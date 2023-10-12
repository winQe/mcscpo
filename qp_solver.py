import numpy as np
from pyomo.environ import (AbstractModel, Var, Objective, Constraint, SolverFactory,
                           maximize, NonNegativeReals, RangeSet)

class QuadraticOptimizer:
    def __init__(self, m):
        self.m = m
        self.model = AbstractModel()
        
        # Index set for nu
        self.model.I = RangeSet(self.m)
        
        # Define variables
        self.model.lambda_var = Var(domain=NonNegativeReals)
        self.model.nu = Var(self.model.I, domain=NonNegativeReals, initialize=0.05)
        
        # Store solutions
        self.optimal_lambda = None
        self.optimal_nu = None

    def solve(self, C, q, r, S, delta):
        # Define the objective
        def objective_rule(mod):
            print("C: ",C)
            print("q: ",q)
            print("r: ",r)
            print("S: ",S)
            print("delta: ",delta)
            nu_np = np.array([mod.nu[i].value for i in mod.I])
            print("nu :",nu_np)

            term1 = -1 / (2 * mod.lambda_var + 1e-8)
            term2 = q + 2 * np.dot(r, nu_np) + np.dot(nu_np, np.dot(S, nu_np))
            term3 = np.dot(C, nu_np)
            term4 = -delta * mod.lambda_var / 2
            return term1 * term2 + term3 + term4
        
        self.model.obj = Objective(rule=objective_rule, sense=maximize)

        # Solve the model
        solver = SolverFactory('ipopt')
        solver.options["tol"] = 1e-8  # Set a tighter tolerance
        solver.options["max_iter"] = 10000  # Increase the maximum number of iterations

        results = solver.solve(self.model,tee=True)

        # Check for infeasibility
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            self.status = "Optimal"
        elif results.solver.termination_condition == 'infeasible':
            self.status = "Infeasible"
        else:
            self.status = "Solver Status: {}".format(results.solver.status)

        # Store solutions
        self.optimal_lambda = self.model.lambda_var.value
        self.optimal_nu = np.array([self.model.nu[i].value for i in self.model.I])

    def get_solution(self):
        if self.status == "Infeasible":
            print("The optimization problem is infeasible!")
            return None, None, self.status
        return self.optimal_lambda, self.optimal_nu, self.status
