import numpy as np
import scipy.sparse.linalg as slg

from typing import Callable, Dict

def _makeJacobianOperator(G : Callable[[np.ndarray, float], np.ndarray],
                          u : np.ndarray, 
                          p : float, 
                          rdiff : float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Constructs a function that calculates the matrix-free Jacobian of G at (u, p) using caching.

    Parameters
    ----------
    G : Callable
        Objective function.
    u : ndarrau
        Current state vector u.
    p : float
        Current parameter value.
    rdiff : float
        Finite-differences step size used to compute the directional derivative.

    Returns
    -------
    jacvec: Callable
        Lambda expression that computes Gu(u, p) * v for any v.
    """
    G_value = G(u, p)
    return lambda v: (G(u + rdiff * v, p) - G_value) / rdiff

def rightmost_eig(G : Callable[[np.ndarray, float], np.ndarray],
                  u : np.ndarray,
                  p : float,
                  sp : Dict) -> float:
    """
    Calcualte the right-most eigenvalue of the Jacobian of G at (u, p). Only returns
    the real part.

    Parameters
    ----------
    G : Callable
        Objective function.
    u : ndarrau
        Current state vector u.
    p : float
        Current parameter value.
    sp : Dict
        Solver parameters.

    Returns
    -------
    real : float
        Real part of the right-most eigenvalue of Gu.
    """

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