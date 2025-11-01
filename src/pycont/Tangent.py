import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt

from .Logger import LOG

from typing import Callable, Dict

def computeTangent(G: Callable[[np.ndarray, float], np.ndarray],
				   u : np.ndarray, 
				   p : float, 
				   prev_tangent : np.ndarray, 
				   sp : Dict,
                   high_accuracy : bool = True) -> np.ndarray:
    """
    Calculate the tangent to the curve defined by G(u,p) = 0 at (u,p) given
    an initial guess `prev_tangent`. This function solves the extended linear
    system [dG/du(u,p) t_u + dG/dp(u,p) t_p; prev_tangent^T (t_u, t_p)] = 0
    using an iterative solver and finite differences approximations of the 
    Jacobian dG/du.

    Parameters
    ----------
    G : callable
        Function representing the nonlinear system, with signature
        ``G(u, p) -> ndarray`` where `u` is the state vector and `p`
        is the continuation parameter.
    u : ndarray
        Current u-point on the curve.
    p : float
        Current parameter value along the curve.
    prev_tangent : ndarray
        Initial guess used for the L-GMRES solver.
    sp : Dict
        Solver parameters.
    high_accuracy : bool (default True)
        If true, a subsequent Newton-Krylov solve is performed when L-GMRES returns
        with insufficient accuracy. scipy.newton_krylov generally calls L-GMRES
        better than I can. Typically `False` for limit cycle continuation.
    
    Returns
    -------
    tangent : ndarray
        Tangent to the curve at (u,p).
    """

    rdiff = sp["rdiff"]
    M = len(u)

    # Create the linear system and right-hand side
    Gp = (G(u, p + rdiff) - G(u, p - rdiff)) / (2.0*rdiff)
    def matvec(v):
        norm_v = lg.norm(v[0:M])
        J = Gp * v[M]
        if norm_v != 0.:
            eps = rdiff / norm_v
            J += (G(u + eps * v[0:M], p) - G(u - eps * v[0:M], p)) / (2.0*eps)
        eq_2 = np.dot(prev_tangent, v)
        return np.append(J, eq_2)
    sys = slg.LinearOperator((M+1, M+1), matvec)
    rhs = np.zeros(M+1); rhs[M] = 1.0

	# Solve the linear system and do postprocessing
    tangent, info = slg.lgmres(sys, rhs, x0=prev_tangent, maxiter=min(M+2, 10))
    tangent_residual = lg.norm(sys(tangent) - rhs)
    LOG.verbose(f'Tangent LGMRES Residual {tangent_residual}, {info}')
    if high_accuracy and tangent_residual > 0.01:
        # Solve the linear system using Newton-Krylov with much better lgmres arguments
        def F(v):
            return matvec(v) - rhs
        tangent = opt.newton_krylov(F, prev_tangent, rdiff=rdiff, verbose=False)
        tangent_residual = lg.norm(F(tangent))
        LOG.verbose(f'Tangent Newton-Krylov Residual {tangent_residual}')

    # Make sure the new tangent lies in the direction of the previous one and return
    tangent = np.sign(np.dot(tangent, prev_tangent)) * tangent / lg.norm(tangent)
    return tangent