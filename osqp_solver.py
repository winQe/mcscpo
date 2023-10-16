import numpy as np
import scipy.sparse as sp
import osqp

import unittest

class QuadraticOptimizerOSQP:
    def __init__(self, m):
        self.m = m
        self.status = None

    def solve(self, C, q, r, S, delta):
        # Construct QP problem matrices
        P = np.block([[S, np.zeros((self.m, 1))],
                      [np.zeros((1, self.m)), -delta/2]])
        P_sparse = sp.csc_matrix(P)

        
        q_vec = np.concatenate([-r, [q]])
        l = np.zeros(self.m + 1)
        u = np.inf * np.ones(self.m + 1)

        # Create an OSQP object
        prob = osqp.OSQP()

        A = sp.csc_matrix(np.eye(self.m + 1))  # Identity matrix of size n x n
        # Setup workspace and change alpha parameter
        prob.setup(P=P_sparse, q=q_vec, A=A, l=l, u=u, alpha=1.0, verbose=True, max_iter=10000)

        # Solve problem
        results = prob.solve()

        # Store solutions
        self.optimal_lambda = results.x[-1]
        self.optimal_nu = results.x[:-1]
        self.status = results.info.status

    def get_solution(self):
        if "solved" not in self.status:
            print("The optimization problem is infeasible!")
            return None, None, self.status
        return self.optimal_lambda, self.optimal_nu, self.status
    
class TestQuadraticOptimizerOSQP(unittest.TestCase):

    def test_solve(self):
        # Given
        C = np.array([0.5, 0.5])
        q = 0.15585482
        r = np.array([1.3896, 0.2497])
        S = np.array([[919.6349, -69.7536], [-69.7538, 502.9334]])
        delta = 0.02
        m = len(C)

        # Initialize the QuadraticOptimizerOSQP class
        optimizer = QuadraticOptimizerOSQP(m)

        # When
        optimizer.solve(C, q, r, S, delta)
        optimal_lambda, optimal_nu, status = optimizer.get_solution()

        # Then
        # Here, you should assert the expected values for optimal_lambda, optimal_nu, and status.
        # For this example, I'm just checking if the status is "solved" as I don't have the expected values.
        self.assertIn("solved", status)

if __name__ == "__main__":
    unittest.main()

# # Example usage
# optimizer = QuadraticOptimizerOSQP(m=2)
# C = np.array([0.33333334, 0.47140449])
# q = 0.3358116
# r = np.array([20.311447, 25.591766])
# S = np.array([[5719.2007, 2661.1472], [2661.1426, 8363.125]])
# delta = 0.02
# optimizer.solve(C, q, r, S, delta)
# optimal_lambda, optimal_nu, status = optimizer.get_solution()
# print("Optimal lambda:", optimal_lambda)
# print("Optimal nu:", optimal_nu)
# print("Status:", status)
