import numpy as np
import scipy.linalg as lg
import scipy.optimize as opt

from typing import Callable, List, Tuple, Dict

def _find_all_zeros(f : Callable[[np.ndarray], float]) -> List[np.ndarray]:
    """
    General function that computes the zeros of a function f with
    signature `f([alpha, beta]) -> float` on the unit circle.
    
    Parameters
    ----------
    f : Callable
        The function to solve.
    
    Returns
    -------
        solutions : List[ndarray]
            The (typically four) unit-vector solutions to the homogeneous quadratic system.
    """
    t_range = np.linspace(0.0, 2.0*np.pi, 10**6 + 1)
    
    prev_f_val = f(np.array([1.0, 0.0]))
    solutions = []
    for n in range(1, t_range.size):
        x = np.array([np.cos(t_range[n]), np.sin(t_range[n])])
        f_val = f(x)

        if f_val * prev_f_val <= 0.0:
            solutions.append(x)
        prev_f_val = f_val

    return solutions

def _solveABSystem(a, b, c):
    """
    Simple function that solves the quadratic form 
        a alpha**2 + 2 b alpha beta + c beta**2 = 0.
    
    Parameters
    ----------
    a, b, c : float
        The three homogeneous coefficients.
    
    Returns
    -------
        solutions : List[ndarray]
            The (typically four) unit-vector solutions to the homogeneous quadratic system.
    """
    f = lambda y: a*y[0]**2 + 2*b*y[0]*y[1] + c*y[1]**2
    solutions = _find_all_zeros(f)

    return solutions

# Gu_v takes arguments u, p, v
def _computeCoefficients(Gu_v : Callable[[np.ndarray, float, np.ndarray], np.ndarray], 
                         Gp : Callable[[np.ndarray, float], np.ndarray], 
                         x_singular : np.ndarray, 
                         phi : np.ndarray, 
                         w : np.ndarray, 
                         w_1 : np.ndarray, 
                         M : int, 
                         r_diff : float) -> Tuple[float, float, float]:
    """
    Compute the coefficients a, b, c in the quadratic form 
        alpha**2 a + 2 alpha beta b + beta**2 c.
    See equation () from [] for more details.

    Parameters
    ----------
    Gu : Callable
        Jacobian of G at the bifurcation point. Callable of signature `Gu(v)` where
        `v` is the direction in which to compute the directional derivative
    Gp : ndarray
        Derivative of `G` with respect to the parameter `p` at the bifurcation point.
    x_singular : ndarray
        The bifurcation point
    phi : ndarray
        Unit vector in the nullspace of Gu(x_singular).
    w : ndarray
        Unit vector that solves Gu * w + Gp = 0.
    w_1 : ndarray
        Unit vector in the nullspace of [Gu | Gp]
    M : int
        Size of the state vector u.
    r_diff : float
        Finite-differences step size for the second-order derivatives

    Returns
    -------
    a, b, c : float
        The three second-order coefficients.
    """

    phi_exp = np.append(phi, 0.0)

    # Compute a
    Gu_phi = lambda x: Gu_v(x[0:M], x[M], phi)
    a = np.dot(phi, (Gu_phi(x_singular + r_diff * phi_exp) - Gu_phi(x_singular)) / r_diff)

    # Compute b
    Gx_w = lambda x: Gu_v(x[0:M], x[M], w) + Gp(x[0:M], x[M])
    b = np.dot(phi, (Gx_w(x_singular + r_diff * phi_exp) - Gx_w(x_singular)) / r_diff)

    # Compute c
    c = np.dot(phi, (Gx_w(x_singular + r_diff * w_1) - Gx_w(x_singular)) / r_diff)

    return a, b, c

# Minimizing the residual of a system is more stable than finding the exact nullspace
def _computeNullspace(Gu : Callable[[np.ndarray], np.ndarray], 
                      Gp : np.ndarray, 
                      M : int, 
                      r_diff : float):
    """
    Compute the nullspaces of the Jacobian Gu and of the extended matrix [Gu | Gp] using
    a minimization formulation rather than a linear solver.

    Parameters
    ----------
    Gu : Callable
        Jacobian of G at the bifurcation point. Callable of signature `Gu(v)` where
        `v` is the direction in which to compute the directional derivative
    Gp : ndarray
        Derivative of `G` with respect to the parameter `p` at the bifurcation point.
    M : int
        Dimension of the state vector `u`
    r_diff : float
        Finite-differences step size for minimization routine.

    Returns
    -------
    phi : ndarray
        Unit nullvector of Gu
    w : ndarray
        Vector that solves Gu * w + Gp = 0
    w_1 : ndarray
        Unit nullvector of [Gu | Gp] obtained by appending a 1 to w.
    """
    phi_0 = np.eye(M)[:,0]
    phi_objective = lambda y: 0.5*np.dot(Gu(y), Gu(y))
    phi_constraint = opt.NonlinearConstraint(lambda y: np.dot(y, y) - 1.0, 0.0, 0.0)
    min_result = opt.minimize(phi_objective, phi_0, constraints=(phi_constraint), options={"eps": r_diff})
    phi = min_result.x

    w_objective = lambda y: np.sqrt(np.dot(Gu(y) + Gp, Gu(y) + Gp))
    min_result = opt.minimize(w_objective, np.zeros(M), method="BFGS", options={"eps": r_diff})
    w = min_result.x
    w_1 = np.append(w, 1.0)
    w_1 = w_1 / lg.norm(w_1)

    return phi, w, w_1

def branchSwitching(G : Callable[[np.ndarray, float], np.ndarray],  
                    x_singular : np.ndarray, 
                    x_prev : np.ndarray, 
                    sp : Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform branch switching at the current bifurcation point `x_singular'.

    Parameters
    ----------
    G : callable
        The objective function.
    x_singular : ndarray
        The bifurcation point (sometimes called a `singular point').
    x_prev : ndarray
        Final point along the branch before the bifurcation point.
    sp : Dict
        Solver parameters.

    Returns
    -------
    directions : List[ndarray]
        List containing the starting points of the new branches, far enough
        away from x_singular.
    tangents : List[ndarray]
        Tangent vectors at the new branches in the starting points.
    """
    print('\nBranch Switching')

    # Setting up variables
    M = x_singular.size - 1
    u = x_singular[0:M]
    p = x_singular[M]

    # Create gradient functions
    rdiff = sp["rdiff"]
    Gu_v = lambda u, p, v: (G(u + rdiff * v, p) - G(u, p)) / rdiff
    Gp = lambda u, p: (G(u, p + rdiff) - G(u, p)) / rdiff

    # Computing necessary coefficients and vectors
    phi, w, w_1 =_computeNullspace(lambda v: Gu_v(u, p, v), Gp(u,p), M, sp["rdiff"])
    a, b, c = _computeCoefficients(Gu_v, Gp, x_singular, phi, w, w_1, M, sp["rdiff"])
    solutions = _solveABSystem(a, b, c)

    # Fina all 4 branch tangents
    directions = []
    tangents = []
    for n in range(len(solutions)):
        alpha = solutions[n][0]
        beta  = solutions[n][1]

        s = 0.01
        N = lambda x: np.dot(alpha*phi + beta/np.sqrt(1.0)*w, x[0:M] - x_singular[0:M]) + beta/np.sqrt(1.0)*(x[M] - x_singular[M]) - s
        F_branch = lambda x: np.append(G(x[0:M], x[M]), N(x))

        tangent = np.append(alpha*phi + beta/np.sqrt(1.0)*w, beta/np.sqrt(1.0))
        x0 = x_singular + s * tangent / lg.norm(tangent)
        dir = opt.newton_krylov(F_branch, x0, rdiff=sp["rdiff"], f_tol=sp["tolerance"])

        directions.append(dir)
        tangents.append(tangent)

    # Remove the direction where we came from
    inner_prodct = -np.inf
    for n in range(len(directions)):
        inner_pd = np.dot(directions[n]-x_singular, x_prev-x_singular) / (lg.norm(directions[n]-x_singular) * lg.norm(x_prev-x_singular))
        if inner_pd > inner_prodct:
            inner_prodct = inner_pd
            idx = n
    directions.pop(idx)
    tangents.pop(idx)
    print('Branch Switching Tangents:', tangents)

    # Returning 3 continuation directions
    return directions, tangents