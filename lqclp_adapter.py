import cvxpy as cp

class LQCLPAdapter:
    def __init__(self, dim_x):
        self.x = cp.Variable(dim_x)
        self.problem = None

    def update_parameters(self, g, b, c, H, delta):
        """Update problem parameters and re-formulate the problem."""
        self.g = g
        self.b = b
        self.c = c
        self.H = H
        self.delta = delta
        
        self.problem = self._formulate_problem()

    def _formulate_problem(self):
        """Formulate the convex optimization problem."""
        
        # Define the objective
        objective = cp.Minimize(self.g.T @ self.x)

        # Define the constraints
        constraints = [
            self.b.T @ self.x + self.c <= 0,
            cp.quad_form(self.x, self.H) <= self.delta
        ]

        # Return the problem
        return cp.Problem(objective, constraints)

    def solve(self):
        """Solve the optimization problem."""
        if self.problem is None:
            raise ValueError("Parameters not set. Use update_parameters() first.")
        return self.problem.solve()

    @property
    def optimal_value(self):
        """Retrieve the optimal value."""
        return self.problem.value

    @property
    def optimal_x(self):
        """Retrieve the optimal x values."""
        return self.x.value
    