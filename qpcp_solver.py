import numpy as np
from scipy.optimize import minimize

class QuadraticOptimizer:
    def __init__(self, m, Hx):
        self.m = m  # number of linear constraints
        self.solution = None
        self.Hx = Hx

    def objective(self, x, g):
        return -g.T @ x # negative for maximization

    def linear_constraints(self, x, B, C):
        return -C - B @ x

    def quad_constraint(self, x, delta):
        return delta - 0.5 * x.T @ self.Hx(x)

    def solve(self, g, B, C, delta):
        n = len(g)
        
        # Linear constraints
        lin_constraints = [{'type': 'ineq', 'fun': lambda x, B=B, C=C, i=i: self.linear_constraints(x, B[i], C[i])} for i in range(self.m)]
        
        # Quadratic constraint
        quad_cons = {'type': 'ineq', 'fun': lambda x: self.quad_constraint(x, delta)}

        constraints = lin_constraints + [quad_cons]
        
        initial_guess = np.zeros(n)
        result = minimize(self.objective, initial_guess, args=(g), constraints=constraints, method='SLSQP', options={'ftol': 1e-9, 'disp': True})
        
        if result.success:
            self.solution = result.x
        else:
            print("Optimization failed!")
            self.solution = None

    def get_solution(self):
        return self.solution

import unittest

class TestQuadraticOptimizer(unittest.TestCase):

    def setUp(self):
        Hx_lambda = lambda x: np.dot(np.array([[1.0, 0.0], [0.0, 1.0]]), x)
        self.optimizer = QuadraticOptimizer(m=2, Hx=Hx_lambda)

    def test_solver(self):
        # Test data
        g = np.array([1.0, 1.0])
        B = np.array([[1.0, 1.0], [-1.0, -1.0]])
        C = np.array([-1.0, 1.0])
        delta = 1.0

        # Solve the optimization problem
        self.optimizer.solve(g, B, C, delta)
        solution = self.optimizer.get_solution()

        # Asserts
        self.assertIsNotNone(solution)  # Solution exists
        self.assertTrue(np.all(self.optimizer.linear_constraints(solution, B[0], C[0]) >= 0))  # First linear constraint satisfied
        self.assertTrue(np.all(self.optimizer.linear_constraints(solution, B[1], C[1]) >= 0))  # Second linear constraint satisfied
        self.assertTrue(self.optimizer.quad_constraint(solution, delta) >= 0)  # Quadratic constraint satisfied

if __name__ == '__main__':
    unittest.main()
