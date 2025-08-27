import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt

from . import PseudoArclengthContinuation as pac
from . import BranchSwitching as brs

from typing import Callable, Optional, Dict, List, Any

class ContinuationResult:
    """Holds continuation output."""
    def __init__(self):
        self.branches : List[Dict[str, np.ndarray]] = []
        self.bifurcation_points : List[np.ndarray] = []

def pseudoArclengthContinuation(G : Callable[[np.ndarray, float], np.ndarray], 
                                u0 : np.ndarray,
                                p0 : float, 
                                ds_min : float, 
                                ds_max : float, 
                                ds_0 : float, 
                                n_steps : int, 
                                solver_parameters : Optional[Dict] = None) -> ContinuationResult:
    """
    Perform pseudo-arclength continuation for a nonlinear system G(u, p) = 0.

    This method numerically tracks solution branches of parameter-dependent
    nonlinear equations using the pseudo-arclength continuation method with 
    internal Newton-Krylov solver. It adapts the step size to remain within 
    the specified bounds and applies Newton iterations at each step to maintain accuracy.

    Parameters
    ----------
    G : callable
        Function representing the nonlinear system, with signature
        ``G(u, p) -> ndarray`` where `u` is the state vector and `p`
        is the continuation parameter.
    u0 : ndarray
        Initial solution vector corresponding to the starting parameter `p0`.
    p0 : float
        Initial value of the continuation parameter.
    ds_min : float
        Minimum allowable continuation step size.
    ds_max : float
        Maximum allowable continuation step size.
    ds_0 : float
        Initial continuation step size.
    n_steps : int
        Maximum number of continuation steps to perform.
    solver_parameters : dict, optional
        Tuning knobs for the corrector and numerics. Recognized keys:
        - "rdiff": float (default 1e-8)
            Finite-difference increment for Jacobian-vector products.
        - "nk_maxiter": int (default 10)
            Maximum Newton-Krylov iterations per corrector.
        - "tolerance": float (default 1e-10)
            Nonlinear residual tolerance for convergence.
        - "bifurcation_detection" : bool (default True)
            Disabling bifurcation detection can significantly speed up continuation when there are no bifurcation points.

    Returns
    -------
    ContinuationResult
        With fields:
        - branches : list of dicts with keys {"u", "p"} holding arrays along each branch
        - bifurcation_points : list of singular points in R^{M+1}

    Notes
    -----
    - Uses forward finite differences for directional derivatives by default.
    - This implementation supports adaptive step size control.
    - The continuation may detect and pass through folds, depending on the
      predictor-corrector scheme implemented.
    - The method is robust for smooth solution branches but may require
      tuning of `tolerance` and `ds_min` for problems with sharp turns
      or bifurcations.
    - Ensure u0 is a converged solution of G(u, p0)=0 for best reliability.
    """
    
    # Verify and set default the solver parameters
    sp = {} if solver_parameters is None else dict(solver_parameters) # shallow copy to avoid changing the user's dict
    rdiff = sp.setdefault("rdiff", 1e-8)
    nk_maxiter = sp.setdefault("nk_maxiter", 10)
    tolerance = sp.setdefault("tolerance", 1e-10)
    sp.setdefault("bifurcation_detection", True)

    # Create gradient functions
    Gu_v = lambda u, p, v: (G(u + rdiff * v, p) - G(u, p)) / rdiff
    Gp = lambda u, p: (G(u, p + rdiff) - G(u, p)) / rdiff

    # Compute the initial tangent to the curve using the secant method
    print('\nComputing Initial Tangent to the Branch.')
    M = u0.size
    u1 = opt.newton_krylov(lambda uu: G(uu, p0 + rdiff), u0, f_tol=tolerance, rdiff=rdiff, maxiter=nk_maxiter)
    initial_tangent = (u1 - u0) / rdiff
    initial_tangent = np.append(initial_tangent, 1.0); initial_tangent = initial_tangent / lg.norm(initial_tangent)
    tangent = pac.computeTangent(u0, p0, Gu_v, Gp, initial_tangent, M, tolerance)

    # Do continuation in both directions of the tangent
    result = ContinuationResult()
    _recursiveContinuation(G, Gu_v, Gp, u0, p0,  tangent, M, ds_min, ds_max, ds_0, n_steps, sp, result)
    _recursiveContinuation(G, Gu_v, Gp, u0, p0, -tangent, M, ds_min, ds_max, ds_0, n_steps, sp, result)

    # Return all found branches and bifurcation points
    return result

def _recursiveContinuation(G : Callable[[np.ndarray, float], np.ndarray], 
                           Gu_v : Callable[[np.ndarray, float, np.ndarray], np.ndarray], 
                           Gp : Callable[[np.ndarray, float], np.ndarray], 
                           u0 : np.ndarray, 
                           p0 : float, 
                           tangent : np.ndarray, 
                           M : int, 
                           ds_min : float, 
                           ds_max : float, 
                           ds : float, 
                           n_steps : int, 
                           sp : Dict[str, Any], 
                           result : ContinuationResult) -> None:
    print('\n\nContinuation on Branch', len(result.branches) + 1)
    
    # Do regular continuation on this branch
    u_path, p_path, bf_points = pac.continuation(G, Gu_v, Gp, u0, p0, tangent, ds_min, ds_max, ds, n_steps, sp)
    result.branches.append({'u': u_path, 'p': p_path})

    # If there are no bifurcation points on this path, return
    if len(bf_points) == 0:
       return
    
    # If there are bifurcation points, check if it is unique
    x_singular = bf_points[0]
    for n in range(len(result.bifurcation_points)):
        if lg.norm(x_singular - result.bifurcation_points[n]) / M < 1.e-4:
            return
    result.bifurcation_points.append(x_singular)
    print('Bifurcation Point at', x_singular)
        
    # The bifurcation point is unique, do branch switching
    x_prev = np.append(u_path[-10,:], p_path[-10]) # x_prev just needs to be a point on the previous path close to the bf point
    directions, tangents = brs.branchSwitching(G, Gu_v, Gp, x_singular, x_prev, sp)

    # For each of the branches, run pseudo-arclength continuation
    for n in range(len(directions)):
        x0 = directions[n]
        _recursiveContinuation(G, Gu_v, Gp, x0[0:M], x0[M], -tangents[n], M, ds_min, ds_max, ds, n_steps, sp, result)