import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, maximize, NonNegativeReals

class QuadraticOptimizer:
    def __init__(self, m):
        self.m = m
        self.model = ConcreteModel()
        
        # Define variables
        self.model.lambda_var = Var(domain=NonNegativeReals)
        self.model.nu = Var(domain=NonNegativeReals)
        
        # Store solutions
        self.optimal_lambda = None
        self.optimal_nu = None

    def solve(self, C, q, r, S, delta):
        # Define the objective
        def objective_rule(mod):
            return (-1 / (2 * mod.lambda_var + 1e-8) * (q + 2 * r * mod.nu + mod.nu ** 2 * S) 
                    + mod.nu * C - (delta * mod.lambda_var / 2))
        
        self.model.obj = Objective(rule=objective_rule, sense=maximize)

        # Solve the model
        solver = SolverFactory('ipopt')
        results = solver.solve(self.model)

        # Check for infeasibility
        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            self.status = "Optimal"
        elif results.solver.termination_condition == 'infeasible':
            self.status = "Infeasible"
        else:
            self.status = "Solver Status: {}".format(results.solver.status)

        # Store solutions
        self.optimal_lambda = self.model.lambda_var.value
        self.optimal_nu = self.model.nu.value

    def get_solution(self):
        if self.status == "Infeasible":
            print("The optimization problem is infeasible!")
            return None, None, self.status
        return self.optimal_lambda, self.optimal_nu, self.status
