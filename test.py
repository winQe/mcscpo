import numpy as np
from scipy.sparse.linalg import cg

def conjugate_gradient_scipy(A, b, x0=None, tol=1e-10, max_iter=1000):
    m, n = b.shape
    if x0 is None:
        x0 = np.zeros((m, n))
    
    x = np.zeros((m, n))
    for i in range(n):
        x[:, i], _ = cg(A, b[:, i], x0=x0[:, i], tol=tol, maxiter=max_iter)
    return x

# Example usage:
A = np.array([[4, 1], [1, 3]])
b = np.array([[1, 2], [2, 3]])
x0 = np.array([[2, 1], [1, 2]])
x = conjugate_gradient_scipy(A, b, x0)
print(x)