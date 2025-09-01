import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt

from . import PseudoArclengthContinuation as pac
from . import BranchSwitching as brs

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any

@dataclass
class ContinuationResult:
    branches: List[pac.Branch] = field(default_factory=list)
    events: List[pac.Event] = field(default_factory=list)

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
    starting_event = pac.Event("SP", u0, p0)
    result.events.append(starting_event)
    _recursiveContinuation(G, Gu_v, Gp, u0, p0,  tangent, M, ds_min, ds_max, ds_0, n_steps, sp, 0, result)
    _recursiveContinuation(G, Gu_v, Gp, u0, p0, -tangent, M, ds_min, ds_max, ds_0, n_steps, sp, 0, result)

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
                           from_event : int,
                           result : ContinuationResult) -> None:
    """
    Internal function that performs pseudo-arclength continuation on the current branch. When the
    continuation routine returns, this method calls the branch-switching routine in case of a
    bifurcation point. If so, it calls itself recursively on each of the three new branches. 

    Parameters
    ----------
    G : callable
        Function representing the nonlinear system, with signature
        ``G(u, p) -> ndarray`` where `u` is the state vector and `p`
        is the continuation parameter.
    Gu_v : callable
        Function calculating the Jacobian of G using matrix-free directional derivatives, 
        with signature ``Gu_v(u, p, v) -> ndarray`` where `u` is the state vector, `p`
        is the continuation parameter, and `v` is the differentiation direction.
    Gp : callable
        Function calculating the derivative of G with respect to the parameter,
        with signature ``Gp(u, p) -> ndarray`` where `u` is the state vector and `p`
        is the continuation parameter.
    u0 : ndarray
        Initial solution vector corresponding to the starting parameter `p0`.
    p0 : float
        Initial value of the continuation parameter.
    tangent : ndarray
        Tangent to the current branch in (u0, p0)
    M : int
        Size of the state variable u
    ds_min : float
        Minimum allowable continuation step size.
    ds_max : float
        Maximum allowable continuation step size.
    ds : float
        Initial continuation step size.
    n_steps : int
        Maximum number of continuation steps to perform.
    sp : dict
        Additional paramters for PyCont.
    from_event : int
        Index of the event that spawned this event (initially a starting point with index 0).
    result: ContinuationResult
        Object that contains all continued branches and detected bifurcation points.

    Returns
    -------
    Nothing, but `result` is updated with the new branche(s) and possible bifurcation points.
    """
    branch_id = len(result.branches)
    print('\n\nContinuation on Branch', branch_id + 1)
    
    # Do regular continuation on this branch
    branch, termination_event = pac.continuation(G, Gu_v, Gp, u0, p0, tangent, ds_min, ds_max, ds, n_steps, branch_id, sp)
    branch.from_event = from_event
    result.branches.append(branch)
    result.events.append(termination_event)
    termination_event_index = len(result.events)-1

    if termination_event.kind != "LP" and termination_event.kind != "BP":
        return

    # If the last point on the previous branch was a fold point, create a new segment where the last one ended.
    elif termination_event.kind == "LP":
        u_final = termination_event.u
        p_final = termination_event.p
        final_tangent = termination_event.info["tangent"]
        _recursiveContinuation(G, Gu_v, Gp, u_final, p_final, final_tangent, M, ds_min, ds_max, ds, n_steps, sp, termination_event_index, result)

    # If there are no bifurcation points on this path, return
    elif termination_event.kind == "BP":
        x_singular = np.append(termination_event.u, termination_event.p)
    
        # If there are bifurcation points, check if it is unique
        for n in range(len(result.events) - 1): # Do not check with yourself
            if result.events[n].kind != "BP":
                continue

            comparison_point = np.append(result.events[n].u, result.events[n].p)
            if lg.norm(x_singular - comparison_point) / M < 1.e-4:
                print('Bifurcation point already discovered. Ending continuation along this branch.')
                return
        
        # The bifurcation point is unique, do branch switching
        x_prev = np.append(branch.u_path[-10,:], branch.p_path[-10]) # x_prev just needs to be a point on the previous path close to the bf point
        directions, tangents = brs.branchSwitching(G, Gu_v, Gp, x_singular, x_prev, sp)

        # For each of the branches, run pseudo-arclength continuation
        for n in range(len(directions)):
            x0 = directions[n]
            _recursiveContinuation(G, Gu_v, Gp, x0[0:M], x0[M], tangents[n], M, ds_min, ds_max, ds, n_steps, sp, termination_event_index, result)