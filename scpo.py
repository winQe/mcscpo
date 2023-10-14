import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import copy
import scpo_core as core
import scipy_solver as solver
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from  safe_rl_envs.envs.engine import Engine as  safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration
import os.path as osp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class SCPOBuffer:
    """
    A buffer for storing trajectories experienced by a SCPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, num_constraints, gamma=0.99, lam=0.95, cgamma=1., clam=0.95):
        self.obs_buf      = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf      = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf      = np.zeros(size, dtype=np.float32)
        self.rew_buf      = np.zeros(size, dtype=np.float32)
        self.ret_buf      = np.zeros(size, dtype=np.float32)
        self.val_buf      = np.zeros(size, dtype=np.float32)
        self.cost_buf     = np.zeros((size,num_constraints), dtype=np.float32) # D buffer for multi-constraints
        self.cost_ret_buf = np.zeros((size,num_constraints), dtype=np.float32) # D return buffer for multi constraints
        self.cost_val_buf = np.zeros((size,num_constraints), dtype=np.float32) # Vd buffer for multi constraints
        self.adc_buf      = np.zeros((size,num_constraints), dtype=np.float32) # Advantage of cost D multi constraints
        self.logp_buf     = np.zeros(size, dtype=np.float32)
        self.mu_buf       = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logstd_buf   = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam # lam -> for GAE
        self.cgamma, self.clam = cgamma, clam # there is no discount for the cost for MMDP 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, cost, cost_val, mu, logstd):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr]      = obs   # augmented state space
        self.act_buf[self.ptr]      = act   # actions (vector of probabilities)
        self.rew_buf[self.ptr]      = rew   # reward
        self.val_buf[self.ptr]      = val   # value function return at current (s,t)
        self.logp_buf[self.ptr]     = logp
        self.cost_buf[self.ptr]     = cost  # actual cost received at timestep
        self.cost_val_buf[self.ptr] = cost_val # D value (we learn D not Jc here I guess) 
        self.mu_buf[self.ptr]       = mu
        self.logstd_buf[self.ptr]   = logstd
        self.ptr += 1

    def finish_path(self, last_val, last_cost_val):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.vstack((self.cost_buf[path_slice], last_cost_val))
        cost_vals = np.vstack((self.cost_val_buf[path_slice], last_cost_val))
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam) # A
        
        # cost advantage calculation
        cost_deltas = costs[:-1] + self.cgamma * cost_vals[1:] - cost_vals[:-1]
        self.adc_buf[path_slice] = core.discount_cumsum(cost_deltas, self.cgamma * self.clam) # AD
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1] # Actual return
        
        # costs-to-go, targets for the cost value function
        self.cost_ret_buf[path_slice] = core.discount_cumsum(costs, self.cgamma)[:-1] # Actual D
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick, std = 1, mean = 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        # center cost advantage, but don't scale
        adc_mean, adc_std = mpi_statistics_scalar(self.adc_buf)
        self.adc_buf = (self.adc_buf - adc_mean)

        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device),
                    cost_ret=torch.FloatTensor(self.cost_ret_buf).to(device),
                    adc=torch.FloatTensor(self.adc_buf).to(device),
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    mu=torch.FloatTensor(self.mu_buf).to(device),
                    logstd=torch.FloatTensor(self.logstd_buf).to(device))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def get_net_param_np_vec(net):
    """
        Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

def cg(Ax, b, cg_iters=2500):
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax', but for x=0, Ax=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r.T)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r.T)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
        # early stopping 
        if np.linalg.norm(p) < EPS:
            break
    return x

def auto_grad(objectives, net, to_numpy=True):
    """
    Get the gradient of the objectives with respect to the parameters of the network.
    """
    # Ensure objectives is an iterable
    if not isinstance(objectives, (list, tuple)):
        objectives = [objectives]

    gradients = []
    for obj in objectives:
        # If the objective is not scalar, loop over its components
        if obj.numel() > 1:
            grad_components = []
            for i in range(obj.numel()):
                grad_component = torch.autograd.grad(obj[i], net.parameters(), retain_graph=True)
                grad_flat = torch.cat([val.flatten() for val in grad_component], axis=0)
                if to_numpy:
                    grad_flat = grad_flat.detach().cpu().numpy()
                grad_components.append(grad_flat)
            gradients.append(grad_components)
        else:
            grad = torch.autograd.grad(obj, net.parameters(), create_graph=True)
            grad_flat = torch.cat([val.flatten() for val in grad], axis=0)
            if to_numpy:
                grad_flat = grad_flat.detach().cpu().numpy()
            gradients.append(grad_flat)
    if len(gradients) == 1:
        return gradients[0]
    return gradients


def auto_hession_x(objective, net, x):
    """
    Returns 
    """
    jacob = auto_grad(objective, net, to_numpy=False)
    
    return auto_grad(torch.dot(jacob, x), net, to_numpy=True)

def scpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, vcf_lr=1e-3, train_v_iters=80, train_vc_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, target_cost = 0.0, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, 
        backtrack_iters=100, model_save=True, cost_reduction=0, num_constraints=2):
    """
    State-wise Constrained Policy Optimization, 
 
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.
        
        vcf_lr (float): Learning rate for cost value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
            
        train_vc_iters (int): Number of gradient descent steps to take on 
            cost value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
            
        target_cost (float): Cost limit that the agent should satisfy

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        backtrack_coeff (float): Scaling factor for line search.
        
        backtrack_iters (int): Number of line search steps.
        
        model_save (bool): If saving model.
        
        cost_reduction (float): Cost reduction imit when current policy is infeasible.

        num_constraints (int): Number of constraints to be enforced

    """
    cost_reduction = np.full(num_constraints, cost_reduction)
    model_save=True
    assert len(target_cost) == num_constraints

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn() 
    # Augmented state space here
    obs_dim = (env.observation_space.shape[0]+num_constraints,) # this is especially designed for SCPO, since we require an additional M in the observation space 
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, num_constraints, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = SCPOBuffer(obs_dim, act_dim, local_steps_per_epoch, num_constraints, gamma, lam)
    
    def compute_kl_pi(data, cur_pi):
        """
        Return the sample average KL divergence between old and new policies
        """
        obs, act, adv, logp_old, mu_old, logstd_old = data['obs'], data['act'], data['adv'], data['logp'], data['mu'], data['logstd']
        # Average KL Divergence  
        # pi, logp = cur_pi(obs, act)
        # average_kl = (logp_old - logp).mean()
        average_kl = cur_pi._d_kl(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(mu_old, dtype=torch.float32),
            torch.as_tensor(logstd_old, dtype=torch.float32), device=device)
        
        return average_kl
    
    def compute_cost_pi(data, cur_pi):
        """
        Return the suggorate cost for current policy
        """
        obs, act, adc, logp_old = data['obs'], data['act'], data['adc'], data['logp']
        
        # Surrogate cost function, D cost not C
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old).unsqueeze(-1)# Make ratio broadcast-compatible with adc
        surr_cost = (ratio * adc).sum(dim=0) #different from cpo one, should still equate to mean
        epochs = len(logger.epoch_dict['EpCost'])
        surr_cost /= epochs # the average 
        
        return surr_cost
        
        
    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for SCPO (SCPO policy loss)
        """
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        # Policy loss 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
        
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    # Set up function for computing cost loss 
    def compute_loss_vc(data, index, vc_net):
        obs, cost_ret = data['obs'], data['cost_ret'][:,index]

        # Split the data into positive and zero cost returns.
        # This is to address potential imbalance in the dataset.
        cost_ret_positive = cost_ret[cost_ret > 0]
        obs_positive = obs[cost_ret > 0]
        cost_ret_zero = cost_ret[cost_ret == 0]
        obs_zero = obs[cost_ret == 0]
        
        if len(cost_ret_zero) > 0:
            # Calculate the fraction of positive returns to zero returns
            frac = len(cost_ret_positive) / len(cost_ret_zero) 
            
            # If there are fewer positive returns than zero returns
            if frac < 1. :# Fraction of elements to keep
                # Randomly down-sample the zero returns to match the number of positive returns.
                indices = np.random.choice(len(cost_ret_zero), size=int(len(cost_ret_zero)*frac), replace=False)
                cost_ret_zero_downsample = cost_ret_zero[indices]
                obs_zero_downsample = obs_zero[indices]
                
                # Combine the positive and down-sampled zero returns
                obs_downsample = torch.cat((obs_positive, obs_zero_downsample), dim=0)
                cost_ret_downsample = torch.cat((cost_ret_positive, cost_ret_zero_downsample), dim=0)
            else:
                # If there's no need to down-sample, use the entire dataset
                obs_downsample = obs
                cost_ret_downsample = cost_ret
        else:
            # If there are no zero returns in the dataset, use the entire dataset
            obs_downsample = obs
            cost_ret_downsample = cost_ret
        # Calculate and return the mean squared error loss between the cost network and the actual cost return
        return ((vc_net(obs_downsample) - cost_ret_downsample)**2).mean()

    get_costs = lambda info, constraints: np.array([info[key] for key in constraints])
    get_d = lambda info, constraints: np.array([info[key] for key in constraints if key != 'cost'])

    
    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizers = [Adam(vc.parameters(), lr=vcf_lr) for vc in ac.vcs]

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # log the loss objective and cost function and value function for old policy
        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        surr_cost_old = compute_cost_pi(data, ac.pi).detach().cpu().numpy()
        # surr_cost_old = surr_cost_old.item()
        v_l_old = compute_loss_v(data).item()

        # SCPO policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        surr_cost = compute_cost_pi(data, ac.pi)
        
        # get Hessian for KL divergence
        kl_div = compute_kl_pi(data, ac.pi)
        
        # Compute dot product of Hessian matrix with x
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        
        # linearize the loss objective and cost function
        g = auto_grad(loss_pi, ac.pi) # get the loss flatten gradient evaluted at pi old 
        B = np.array(auto_grad(surr_cost, ac.pi)) # get the cost increase flatten gradient evaluted at pi old
        # get the Episode cost
        EpLen = logger.get_stats('EpLen')[0]
        # should be an array, not right
        EpMaxCost = logger.get_stats('EpMaxCost') #EpMaxCost = M, different compared to EpCost
        assert len(EpMaxCost) == 2
        # cost constraint linearization
        '''
        original fixed target cost, in the context of mean adv of epochs
        '''
        # c = EpMaxCost - target_cost 
        # rescale  = EpLen
        # c /= (rescale + EPS)
        
        '''
        fixed target cost, in the context of sum adv of epoch
        '''
        # if negative means M (maximum statewise cost) cost limit, thus infeasible
        c = np.array(EpMaxCost) - np.array(target_cost)

        # core calculation for SCPO
        # Conjugate Gradient to calculate H^-1
        Hinv_g   = cg(Hx, g)             # Hinv_g = H^-1 * g        
        approx_g = Hx(Hinv_g)           # g
        # print("g approximation error ", np.linalg.norm(approx_g - g))
        # q        = np.clip(Hinv_g.T @ approx_g, 0.0, None)  # g.T / H @ g
        # Analytical solution from the CPO paper (Appendix 10.2)
        # q = g.T * H^-1 * g
        q        = Hinv_g.T @ approx_g

        # Hinv_b1 = cg(Hx,B[0])
        # Hinv_b2 = cg(Hx,B[1])
        # approx_b1 = Hx(Hinv_b1)
        # approx_b2 = Hx(Hinv_b1)
        # print("b1 approximation error ", np.linalg.norm(approx_b1 - B[0]))
        # print("b2 approximation error ", np.linalg.norm(approx_b2 - B[1]))

        Hinv_B = np.array([cg(Hx, b) for b in B])
        approx_B = np.array([Hx(Hinv_b) for Hinv_b in Hinv_B])
        # def verify_cg(Hx, B, Hinv_B, approx_B):
        #     # Compute the residual for each column of B using Hx
        #     residuals = [Hx(Hinv_b) - b for Hinv_b, b in zip(Hinv_B, B)]
            
        #     # Compute the norm of each residual
        #     residual_norms = [np.linalg.norm(res) for res in residuals]
            
        #     # Check if the approximated B is close to the original B
        #     approx_error = np.linalg.norm(approx_B - B)
            
        #     # Print the results
        #     for i, norm in enumerate(residual_norms):
        #         print(f"Residual norm for column {i+1}: {norm:.2e}")
            
        #     print(f"Approximation error: {approx_error:.2e}")

        #     # Return True if all residuals are close to zero and the approximation error is small
        #     return all(norm < 1e-6 for norm in residual_norms) and approx_error < 1e-6

        # # Assuming Hx, B, Hinv_B, and approx_B are already defined
        # is_correct = verify_cg(Hx, B, Hinv_B, approx_B)
        # print("Implementation is correct:" if is_correct else "Implementation might be incorrect.")
        r = Hinv_B @ approx_g          # b^T H^{-1} g
        S =  approx_B @ Hinv_B.T      # b^T H^{-1} b
        # S is supposed to be symmetric, something not right here

        # A = q - r**2 / s            # should be always positive (Cauchy-Shwarz)
        # # whether or not the plane of the linear constraint intersects the quadratic trust region, CPO paper appendix
        # B = 2*target_kl - c**2 / s  # does safety boundary intersect trust region? (positive = yes)
        
        # # solve QP
        # # decide optimization cases (feas/infeas, recovery)
        # # Determine optim_case (switch condition for calculation,
        # # based on geometry of constrained optimization problem)
        # paper_timer = time.time()
        # if np.any(b.T @ b <= 1e-8) and np.all(c < 0):
        #     # cost grad is zero
        #     # all area in trust region satisfy the constraints, don't need to consider constraints for this optimization
        #     Hinv_b, r, s, A, B = 0, 0, 0, 0, 0
        #     optim_case = 4
        # else:
        #     # cost grad is nonzero: SCPO update!
        #     import ipdb; ipdb.set_trace()
        #     Hinv_b = cg(Hx, b)                # H^{-1} b
        #     r = Hinv_b.T @ approx_g          # b^T H^{-1} g
        #     s = Hinv_b.T @ Hx(Hinv_b)        # b^T H^{-1} b
        #     A = q - r**2 / s            # should be always positive (Cauchy-Shwarz)
        #     # whether or not the plane of the linear constraint intersects the quadratic trust region, CPO paper appendix
        #     B = 2*target_kl - c**2 / s  # does safety boundary intersect trust region? (positive = yes)

        #     # c < 0: feasible

        #     if c < 0 and B < 0:
        #         # point in trust region is feasible and safety boundary doesn't intersect
        #         # ==> entire trust region is feasible
        #         # If c2=s − δ > 0 and c < 0, then the quadratic trust region lies entirely within the linear constraint-satisfying halfspace,
        #         # and we can remove the linear constraint without changing the optimization problem
        #         optim_case = 3
        #     elif c < 0 and B >= 0:
        #         # x = 0 is feasible and safety boundary intersects
        #         # ==> most of trust region is feasible
        #         optim_case = 2
        #     elif c >= 0 and B >= 0:
        #         # x = 0 is infeasible and safety boundary intersects
        #         # ==> part of trust region is feasible, recovery possible
        #         optim_case = 1
        #         print(colorize(f'Alert! Attempting feasible recovery!', 'yellow', bold=True))
        #     else:
        #         # x = 0 infeasible, and safety halfspace is outside trust region
        #         # ==> whole trust region is infeasible, try to fail gracefully
        #         optim_case = 0
        #         print(colorize(f'Alert! Attempting INFEASIBLE recovery!', 'red', bold=True))
        
        # print(colorize(f'optim_case: {optim_case}', 'magenta', bold=True))
        
        
        # # get optimal theta-theta_k direction
        # if optim_case in [3,4]:
        #     # all area in trust region satisfy the constraints, don't need to consider constraints for this optimization
        #     lam = np.sqrt(q / (2*target_kl))
        #     nu = 0
        # elif optim_case in [1,2]:
        #     # bounds
        #     LA, LB = [0, r /c], [r/c, np.inf]
        #     LA, LB = (LA, LB) if c < 0 else (LB, LA)
        #     # do the projection
        #     proj = lambda x, L : max(L[0], min(L[1], x))
        #     lam_a = proj(np.sqrt(A/B), LA)
        #     lam_b = proj(np.sqrt(q/(2*target_kl)), LB)
        #     f_a = lambda lam : -0.5 * (A / (lam+EPS) + B * lam) - r*c/(s+EPS)
        #     f_b = lambda lam : -0.5 * (q / (lam+EPS) + 2 * target_kl * lam)
        #     lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
        #     # nu = max(0, lam * c - r) / (np.clip(s,0.,None)+EPS)
        #     nu = max(0, lam * c - r) / (s+EPS)
        # else:
        #     lam = 0
        #     # nu = np.sqrt(2 * target_kl / (np.clip(s,0.,None)+EPS))
        #      # decrease the constraint value, infeasibility recovery
        #     nu = np.sqrt(2 * target_kl / (s+EPS))
            
        # lam_paper = lam
        # nu_paper = nu
        # # normal step if optim_case > 0, but for optim_case =0,
        # # perform infeasible recovery: step to purely decrease cost
        # # step in the paper
        # # need to handle for multiple constraints as well, think about how???
        # x_direction_paper = (1./(lam+EPS)) * (Hinv_g + nu * Hinv_b) if optim_case > 0 else nu * Hinv_b

        # print("lambda and nu value from paper = [{},{}]".format(lam_paper,nu_paper))
        # paper_end = time.time()
        # paper_time = paper_end - paper_timer
        # print(f"Time taken: {paper_time} seconds")


        # Start timing
        solver_start = time.time()

        # Use QP library to solve the QP
        # Setup QP Solver
        qp_solver = solver.QuadraticOptimizer(num_constraints)
        qp_solver.solve(c,q,r,S,target_kl)
        lam,nu,status = qp_solver.get_solution()
        if status == "Infeasible":
            # decrease the constraint value, infeasibility recovery
            #
            breakpoint()
            # First, check which constraints are infeasible
            # Infeasible if c > 0 and c**2/s − δ > 
            # (the intersection of the quadratic trust region and linear constraint-satisfying halfspace is empty)
            
            s = np.array([approx_B[i] @ Hinv_B[i] for i in range(approx_B.shape[0])])
            # if element = 1, corresponding constraint is inseasible, 0 means it's not
            infeasibility = infeasibility = np.where(((c**2/s - target_kl > 0) & (c > 0)), 1, 0)
            nu = np.sqrt(2 * target_kl / (s+EPS)) * infeasibility
            x_direction = np.sum(nu[:, np.newaxis] * Hinv_B, axis=1)
        else:
            x_direction = (1./(lam+EPS)) * (Hinv_g + nu @ Hinv_B) 
        
        # Stop timing
        solver_end = time.time()

        np.set_printoptions(precision=4, suppress=True)
        print("lambda and nu value from solver = [{},{}]".format(lam,nu))
        print(f"Time taken: {solver_end - solver_start} seconds")

        # # Quantitative Comparison
        # # L2 distance
        # l2_distance = np.linalg.norm(x_direction_paper - x_direction)
        # print(f"L2 distance between methods: {l2_distance}")

        # # Cosine similarity
        # dot_product = np.dot(x_direction_paper, x_direction)
        # norm_product = np.linalg.norm(x_direction_paper) * np.linalg.norm(x_direction)
        # cosine_similarity = dot_product / (norm_product + EPS)
        # print(f"Cosine similarity between methods: {cosine_similarity}")

        # # L1 difference
        # l1_difference = np.mean(np.abs(x_direction_paper - x_direction))
        # print(f"Mean absolute difference between methods: {l1_difference}")

        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) - step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            surr_cost = compute_cost_pi(data, actor_tmp)
            
            return kl, pi_l, surr_cost
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        # backtracking line search to enforce constraint satisfaction
        for j in range(backtrack_iters):
            try:
                kl, pi_l_new, surr_cost_new = set_and_eval(backtrack_coeff**j)
            except:
                import ipdb; ipdb.set_trace()

            if (kl.item() <= target_kl and
                (pi_l_new.item() <= pi_l_old if status != "Infeasible" else True) and # if current policy is feasible (optim>1), must preserve pi loss
                np.all((surr_cost_new.detach().cpu().numpy() - surr_cost_old) <= np.maximum(-c,-cost_reduction))):
                
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                
                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                
                loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
                surr_cost = compute_cost_pi(data, ac.pi) # re-evaluate the surr_cost for the new policy
                break
            if j==backtrack_iters-1:
                print(colorize(f'Line search failed! Keeping old params.', 'yellow', bold=False))

        # Value function learning
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # Cost value function learning
        for _ in range(train_vc_iters):
            index = 0
            for optimizer, vc_net in zip(vcf_optimizers, ac.vcs):
                optimizer.zero_grad()
                loss_vc = compute_loss_vc(data,index,vc_net)  # Assuming compute_loss_vc can take a specific vc as an argument
                loss_vc.backward()
                mpi_avg_grads(vc_net)  # average grads across MPI processes for the specific vc
                optimizer.step()
                index += 1

        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old, LossCost=surr_cost_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old),
                     DeltaLossCost=(surr_cost.detach().cpu().numpy() - surr_cost_old))

    # Prepare for interaction with environment
    start_time = time.time()
    
    while True:
        try:
            o, ep_ret, ep_len = env.reset(), 0, 0
            break
        except:
            print('reset environment is wrong, try next reset')
    
    # Initialize the environment and cost all = 0
    ep_cost_ret = np.zeros(num_constraints + 1, dtype=np.float32)
    ep_cost = np.zeros(num_constraints + 1, dtype=np.float32)
    cum_cost = 0
    M = np.zeros(num_constraints, dtype=np.float32) # initialize the maximum cost a 0 per constraints
    o_aug = np.append(o, M) # augmented observation = observation + M 
    first_step = True
    constraints_list = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # Forward, get action and value estimates (cost and reward) for the current observation
            a, v, vcs, logp, mu, logstd = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
            
            try: 
                next_o, r, d, info = env.step(a)
                assert 'cost' in info.keys()
            except: 
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
                info['cost'] = 0 # no cost when episode done    
            
            if first_step:
                # the first step of each episode
                constraints_list = [key for key in info.keys()]
                cost_increase =  get_d(info, constraints_list)
                # cost_increase = info['cost'] # define the new observation and cost for Maximum Markov Decision Process
                M_next = cost_increase
                first_step = False
            else:
                # the second and forward step of each episode
                # cost increase = D, to be constrained to ensure state-wise safety, constraining maximum violation in state transition -> enforcing statewise safety
                costs_D = np.array(get_d(info, constraints_list))
                cost_increase = np.maximum(costs_D - M, 0)
                M_next = M + cost_increase
             
            # Track cumulative cost over training
            # TODO: log each cost (cost1 cost2 cost3)
            cum_cost += info['cost'] # not equal to M
            ep_ret += r
            ep_cost_ret += get_costs(info, constraints_list) * (gamma ** t)
            ep_cost += get_costs(info, constraints_list)
            ep_len += 1

            # save and log, buffer is different, store 
            buf.store(o_aug, a, r, v, logp, cost_increase, vcs, mu, logstd)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            # o = next_o
            M = M_next
            o_aug = np.append(next_o, M_next)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _, _ = ac.step(torch.as_tensor(o_aug, dtype=torch.float32))
                    vc = np.zeros(num_constraints,dtype=np.float32)
                else:
                    v = 0
                    vc = np.zeros(num_constraints,dtype=np.float32)

                buf.finish_path(v, vc)
                if terminal:
                    # only save EpRet / EpLen / EpCostRet if trajectory finished
                    # EpMaxCost = Max MDP M, while EpCost is just CMDP, it's Maximum state-wise cost, cannot be canceled out with negative cost if that even exist
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCostRet=ep_cost_ret, EpCost=ep_cost, EpMaxCost=M)
                while True:
                    try:
                        o, ep_ret, ep_len = env.reset(), 0, 0
                        break
                    except:
                        print('reset environment is wrong, try next reset')
                ep_cost_ret = np.zeros(num_constraints + 1, dtype=np.float32)
                ep_cost = np.zeros(num_constraints + 1, dtype=np.float32)
                M = np.zeros(num_constraints, dtype=np.float32) # initialize the maximum cost a 0 per constraints
                o_aug = np.append(o, M) # augmented observation = observation + M 
                first_step = True

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform SCPO update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpCostRet', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('EpMaxCost', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossCost', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('DeltaLossCost', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
        
def create_env(args):
    env =  safe_rl_envs_Engine(configuration(args.task))
    return env

def parse_float_list(s):
    return [float(item) for item in s.split(',')]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Goal_Point_8Hazards')
     
    parser.add_argument('--target_cost', type=parse_float_list, default=[0.00]) # the array of cost limit for the environment
    parser.add_argument('--target_kl', type=float, default=0.02) # the kl divergence limit for SCPO
    parser.add_argument('--cost_reduction', type=float, default=0.) # the cost_reduction limit when current policy is infeasible
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='solverbased_scpo')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--num_constraints', type=int, default=1) # Number of constraints

    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    exp_name = args.task + '_' + args.exp_name \
                + '_' + 'kl' + str(args.target_kl) \
                + '_' + 'target_cost' + str(args.target_cost) \
                + '_' + 'epoch' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False

    scpo(lambda : create_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, target_cost=args.target_cost, 
        model_save=model_save, target_kl=args.target_kl, cost_reduction=args.cost_reduction, num_constraints= args.num_constraints)