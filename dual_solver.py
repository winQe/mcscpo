import pyomo.environ as pyo

class DualSolver:
    def __init__(self, z, C, B, g, delta):
        self.z = z
        self.C = C
        self.B = B
        self.g = g
        self.delta = delta
        self.model = pyo.ConcreteModel()
        self._setup_model()

    def _setup_model(self):
        # Define the dual variables
        self.model.nu = pyo.Var(range(len(self.C)), domain=pyo.NonNegativeReals)
        self.model.lambda_ = pyo.Var(domain=pyo.NonNegativeReals)
        
        # Define the dual function
        self.model.obj = pyo.Objective(rule=self._dual_func, sense=pyo.maximize)

    def _dual_func(self, model):
        term1 = sum(model.nu[i] * (self.C[i] + sum(self.B[i][j] * self.z[j] for j in range(len(self.z))) ) for i in range(len(self.C)))
        term2 = sum(self.g[j] * self.z[j] for j in range(len(self.g)))
        term3 = model.lambda_ * (-self.delta + 0.5 * sum((self.g[j] + sum(self.B[i][j] * model.nu[i] for i in range(len(self.C)))) * self.z[j] for j in range(len(self.g))))
        return term1 + term2 + term3

    def solve(self):
        # Solve the model using IPOPT
        solver = pyo.SolverFactory('ipopt')
        self.results = solver.solve(self.model, tee=True)

    def get_results(self):
        # Return the results
        return {
            "Optimal value": self.model.obj(),
            "Optimal nu": [self.model.nu[i].value for i in range(len(self.C))],
            "Optimal lambda": self.model.lambda_.value
        }
