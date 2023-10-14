import numpy as np
from scipy.optimize import minimize

class QuadraticOptimizer:
    def __init__(self, m):
        self.m = m
        self.status = None
        self.optimal_lambda = None
        self.optimal_nu = None

    def objective(self, x, C, q, r, S, delta):
        lambda_val = x[0]
        nu = x[1:]
        # Negate the objective function to maximize
        return 1/(2*lambda_val + 1e-8) * (q + 2*np.dot(r, nu) + np.dot(nu, np.dot(S, nu))) - np.dot(C, nu) + delta*lambda_val/2

    def solve(self, C, q, r, S, delta):
        # Initial guess
        x0 = [0.1] + [0.1] * self.m  # [lambda, nu1, nu2, ...]

        # Constraints: lambda and nu >= 0
        cons = [{'type': 'ineq', 'fun': lambda x: x[i]} for i in range(len(x0))]

        # Solve the optimization problem
        res = minimize(self.objective, x0, args=(C, q, r, S, delta), constraints=cons, method='SLSQP')

        # Update the solution status and values
        if res.success:
            self.status = "Optimal"
            self.optimal_lambda = res.x[0]
            self.optimal_nu = res.x[1:]
        else:
            self.status = "Infeasible"

    def get_solution(self):
        if self.status == "Infeasible":
            print("The optimization problem is infeasible!")
            return None, None, self.status
        return self.optimal_lambda, self.optimal_nu, self.status

import unittest
import numpy as np

class TestQuadraticOptimizer(unittest.TestCase):

    def test_optimization(self):
        # Given values
        C = np.array([0.5, 0.5])
        q = 0.15585482
        r = np.array([1.3896, 0.2497])
        S = np.array([[919.6349, -69.7536], [-69.7538, 502.9334]])
        delta = 0.02
        m = len(C)

        # Create an instance of the QuadraticOptimizer
        optimizer = QuadraticOptimizer(m)
        optimizer.solve(C, q, r, S, delta)
        lambda_val, nu_values, status = optimizer.get_solution()

        # Check if the solution is optimal
        self.assertEqual(status, "Optimal")
        print("lambda and nu value from solver = [{},{}]".format(lambda_val,nu_values))


        # Additional checks can be added based on expected lambda and nu values
        # For example:
        # self.assertAlmostEqual(lambda_val, expected_lambda, places=5)
        # self.assertTrue(np.allclose(nu_values, expected_nu_values, atol=1e-5))

    def test_optimization_with_given_values(self):
        # Given values
        C = np.array([0.33333334, 0.47140449])
        q = 0.10000436
        r = np.array([5.1337132, 7.596346])
        S = np.array([[1245.691, 715.49426], [715.4936, 2107.7969]])
        delta = 0.02
        m = len(C)

        # Create an instance of the QuadraticOptimizer
        optimizer = QuadraticOptimizer(m)
        optimizer.solve(C, q, r, S, delta)
        lambda_val, nu_values, status = optimizer.get_solution()

        # Check if the solution is optimal
        self.assertEqual(status, "Infeasible")


if __name__ == '__main__':
    unittest.main()
