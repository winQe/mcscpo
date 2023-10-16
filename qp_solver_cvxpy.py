import cvxpy as cp
import numpy as np

def ensure_matrix_form(var):
    if np.isscalar(var):
        return np.array([[var]])
    elif isinstance(var, np.ndarray) and var.ndim == 1:
        return var.reshape(-1, 1)  # converts it to column matrix
    else:
        return var

class QuadraticOptimizer:
    def __init__(self, m):
        self.m = m
        
        # Define variables
        self.lambda_var = cp.Variable(pos=True)  # ensuring lambda >= 0
        self.nu = cp.Variable(pos=True)  # ensuring nu >= 0

        # self.nu = cp.Variable((m, 1), nonneg=True) # ensuring nu >= 0
        
        # Store solutions
        self.optimal_lambda = None
        self.optimal_nu = None

    def solve(self,C, q, r, S, delta):
        # Objective function
        # objective = cp.Maximize(
        #     (-1/(2*self.lambda_var) * (q + 2 * r.T @ self.nu + self.nu.T @ S @ self.nu) 
        #     + self.nu.T @ C - (delta*self.lambda_var/(2)))
        # )

        objective = cp.Maximize(
            (-1/(2*self.lambda_var + 1e-8) * (q + 2 * r * self.nu + self.nu **2 * S) 
            + self.nu * C - (delta*self.lambda_var/(2)))
        )
        
        
        # Construct problem and solve
        prob = cp.Problem(objective)
        prob.solve()

        # Check for infeasibility
        if prob.status == cp.INFEASIBLE:
            self.status = "Infeasible"
        else:
            self.status = "Optimal"
        
        # Store solutions
        self.optimal_lambda = self.lambda_var.value
        self.optimal_nu = self.nu.value

    def get_solution(self):
        if self.status == "Infeasible":
            print("The optimization problem is infeasible!")
            return None, None, self.status
        return self.optimal_lambda, self.optimal_nu, self.status
