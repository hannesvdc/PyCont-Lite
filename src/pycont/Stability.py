import numpy as np
import scipy.sparse.linalg as slg

from typing import Callable, Dict

def _makeJacobianOperator(G, u, p, rdiff):
    G_value = G(u, p)
    return lambda v: (G(u + rdiff * v, p) - G_value) / rdiff

def rightmost_eig(G : Callable[[np.ndarray, float], np.ndarray],
                  u : np.ndarray,
                  p : float,
                  sp : Dict) -> float:
    M = len(u)
    rdiff = sp["rdiff"]
    jacobian = _makeJacobianOperator(G, u, p, rdiff)

    # Special case for one-dimensional state vectors - Arnoldi won't work
    if M == 1:
        rightmost_eigenvalue = jacobian(np.array([1.0]))
    else:
        J = slg.LinearOperator((M,M), jacobian)
        eig_vals = slg.eigs(J, k=1, which='LR', return_eigenvectors=False)
        rightmost_eigenvalue = eig_vals[0]

    # Only return the real part
    return float(np.real(rightmost_eigenvalue))