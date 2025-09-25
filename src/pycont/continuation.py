import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt

from . import PseudoArclengthContinuation as pac
from . import BranchSwitching as brs
from . import Stability as stability
from .Types import ContinuationResult, Event, InputError
from .Tangent import computeTangent
from .Logger import LOG, Verbosity, configureLOG

from typing import Callable, Optional, Dict, Any

def _as_1d_float(x, name: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float)
    except Exception as e:
        raise InputError(f"{name} cannot be converted to a float array") from e
    if a.ndim == 0:  # scalar -> length-1
        a = a.reshape(1)
    if a.ndim != 1:
        raise InputError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise InputError(f"{name} contains NaN/Inf")
    return a

def _as_scalar_float(x, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise InputError(f"{name} must be a scalar float") from e
    if not np.isfinite(v):
        raise InputError(f"{name} is NaN/Inf")
    return v

def pseudoArclengthContinuation(G : Callable[[np.ndarray, float], np.ndarray], 
                                u0 : np.ndarray,
                                p0 : float, 
                                ds_min : float, 
                                ds_max : float, 
                                ds_0 : float, 
                                n_steps : int, 
                                solver_parameters : Optional[Dict] = None,
                                verbosity: Verbosity | str | int = Verbosity.INFO,) -> ContinuationResult:
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
        - "analyze_stability" : bool (default True)
            By default, the real part of the leading eigenvalue of Gu is computed. Negative eigenvalue indicates a 
            stable branch, and a positive eigenvalue means the branch is unstable. Stability analysis can be disabled
            to speed up computations.
        - "initial_directions" : str (default 'both')
            Choose whether to explore only increasing or decreasing parameter values by passing 'increase_p' or 
            'decrease_p' respectively. Default is 'both'.
        - "param_min" : float (default None)
            User-speficied minimal allowed parameter value. Continuation will not go lower than this limit.
        - "param_max" : float (default None)
            User-speficied maximal allowed parameter value. Continuation will not go higher than this limit.

    verbosity : Verbosity or String or Int
        The level of verbosity required by the user. Can either be Verbosity.QUIET (1), Verbosity.INFO (2) or Verbosity.VERBOSE (3).
        Any string representation of these words will also be accepted.

    Returns
    -------
    ContinuationResult
        With fields:
        - branches
        - events

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
    u0 = _as_1d_float(u0, "u0")
    p0 = _as_scalar_float(p0, "p0")
    M = len(u0)
    
    # Verify and set default the solver parameters
    sp = {} if solver_parameters is None else dict(solver_parameters)
    rdiff = sp.setdefault("rdiff", 6.6e-6)
    nk_maxiter = sp.setdefault("nk_maxiter", 10)
    tolerance = sp.setdefault("tolerance", 1e-10)
    sp.setdefault("bifurcation_detection", True)
    sp.setdefault("analyze_stability", True)
    mode = sp.setdefault("initial_directions", "both").lower()
    param_min = sp.setdefault("param_min", None)
    param_max = sp.setdefault("param_max", None)
    sp.setdefault("seed", 12345)
    n_bifurcation_vectors = sp.setdefault("n_bifurcation_vectors", min(3, M))
    sp.setdefault("hopf_detection", False)
    r_keep = sp.setdefault("r_keep", 2)
    m_target = sp.setdefault("m_target", 16)

    # Perform basic checks on some parameters without raising an error
    sp["r_keep"] = min(M, r_keep)
    sp["m_target"] = min(m_target, M)

    # Create the logger based on the user's verbosity requirement.
    configureLOG(verbosity=verbosity)

    # Perform necessary chechs on the user's input
    G0 = G(u0, p0)
    if not np.all(np.isfinite(G0)):
        raise InputError(f"Initial function evaluation contains NaN or Inf {G0}.")
    if G0.size != u0.size:
        raise InputError(f"Shape mismatch between u0 and G(u0, p0). Got shapes {u0.shape} and {G0.shape} respectively.")
    if n_steps < 1 or int(n_steps) != n_steps:
        raise InputError(f"n_steps must be a positive integer, got {n_steps}")
    if ds_min <= 0 or ds_0 <= 0 or ds_max <= 0:
        raise InputError(f"ds_0, ds_min, ds_max must be > 0. Got {ds_0}, {ds_min}, and {ds_max}.")
    if ds_max < ds_min:
        raise InputError("ds_max must be >= ds_min")
    ds_0 = float(np.clip(ds_0, ds_min, ds_max))
    if param_min is not None and param_max is not None and param_min >= param_max:
        raise InputError("require param_min < param_max")
    if param_min is not None and p0 < param_min:
        raise InputError("p0 must be inside (param_min, param_max)")
    if param_max is not None and p0 > param_max:
        raise InputError("p0 must be inside (param_min, param_max)")
    if nk_maxiter <= 0:
        raise InputError(f"nk_maxiter must be strictly positive. Got {nk_maxiter}.")
    if tolerance <= 0.0:
        raise InputError(f"tolerance must be strictly positive. Got {tolerance}.")
    if n_bifurcation_vectors < 0:
        raise InputError(f"number of bifurcation vectors must be a positive integer, got {n_bifurcation_vectors}.")
    if m_target > r_keep:
        raise InputError(f"r_keep cannot be larger than m_target for Hopf bifurcation detection, got {r_keep} and {m_target}.")

    # Compute the initial tangent to the curve using the secant method
    LOG.info('\nComputing Initial Tangent to the Branch.')
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        try:
            u1 = opt.newton_krylov(lambda uu: G(uu, p0 + rdiff), u0, f_tol=tolerance, rdiff=rdiff, maxiter=nk_maxiter)
        except opt.NoConvergence:
            raise InputError("Initial tangent computation failed.")
    initial_tangent = (u1 - u0) / rdiff
    initial_tangent = np.append(initial_tangent, 1.0); initial_tangent = initial_tangent / lg.norm(initial_tangent)
    tangent = computeTangent(G, u0, p0, initial_tangent, sp)
    if not np.all(np.isfinite(tangent)):
        raise InputError(f"Initial tangent contains NaN or Inf values {tangent}. Perhaps G(u0, p0) is not finite?")

    # Make a list of which directions to explore (increase_p, decrease_p or both)
    mode = mode.lower()
    if mode == "both" or tangent[M] == 0.0: # Edge case if we start on a fold point
        dirs = [tangent, -tangent]
    elif mode == "increase_p":
        dirs = [tangent if tangent[M] > 0 else -tangent]
    elif mode == "decrease_p":
        dirs = [tangent if tangent[M] < 0 else -tangent]
    else:
        raise ValueError(f"Initial Directions must be 'both', 'increase_p' or 'decrease_p'(got {mode})")

    # Filter the initial directions based on param_min / param_max if they are set
    valid_dirs = []
    for direction in dirs:
        if param_min is not None and p0 <= param_min and direction[M] < 0.0:
            continue
        if param_max is not None and p0 >= param_max and direction[M] > 0.0:
            continue
        valid_dirs.append(direction)
    dirs = valid_dirs

    # Do continuation in both directions of the tangent
    result = ContinuationResult()
    starting_event = Event("SP", u0, p0, 0.0)
    result.events.append(starting_event)
    for t0 in dirs:
        _recursiveContinuation(G, u0, p0, t0, ds_min, ds_max, ds_0, n_steps, sp, 0, result)

    # Return all found branches and bifurcation points
    return result

def _recursiveContinuation(G : Callable[[np.ndarray, float], np.ndarray], 
                           u0 : np.ndarray, 
                           p0 : float, 
                           tangent : np.ndarray, 
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
    u0 : ndarray
        Initial solution vector corresponding to the starting parameter `p0`.
    p0 : float
        Initial value of the continuation parameter.
    tangent : ndarray
        Tangent to the current branch in (u0, p0)
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
    LOG.info(f'\n\nContinuation on Branch {branch_id + 1}')
    
    # Do regular continuation on this branch
    branch, termination_event = pac.continuation(G, u0, p0, tangent, ds_min, ds_max, ds, n_steps, branch_id, sp)
    branch.from_event = from_event
    result.branches.append(branch)
    result.events.append(termination_event)
    termination_event_index = len(result.events)-1

    # Calculate the eigenvalues with largest real part to analyze stability of the branch
    if sp["analyze_stability"]:
        LOG.info("Analyzing stability by computing the right-most eigenvalue...")
        index = len(branch.p_path) // 2
        rightmost_eigenvalue_realpart = stability.rightmost_eig_realpart(G, branch.u_path[index,:], branch.p_path[index], sp)
        branch.stable = (rightmost_eigenvalue_realpart < 0.0)
        LOG.info('Stable' if branch.stable else 'Unstable')

    # If there are no bifurcation or fold points on this path, return
    if termination_event.kind != "LP" and termination_event.kind != "BP":
        return

    # If the last point on the previous branch was a fold point, create a new segment where the last one ended.
    elif termination_event.kind == "LP":
        u_final = termination_event.u
        p_final = termination_event.p
        final_tangent = termination_event.info["tangent"]
        _recursiveContinuation(G, u_final, p_final, final_tangent, ds_min, ds_max, ds, n_steps, sp, termination_event_index, result)

    # If there is a bifurcation point on this path, do branch switching
    elif termination_event.kind == "BP":
        x_singular = np.append(termination_event.u, termination_event.p)
    
        # If there are bifurcation points, check if it is unique
        for n in range(len(result.events) - 1): # Do not check with yourself
            if result.events[n].kind != "BP":
                continue

            comparison_point = np.append(result.events[n].u, result.events[n].p)
            if lg.norm(x_singular - comparison_point) / len(x_singular) < 1.e-4:
                LOG.info('Bifurcation point already discovered. Ending continuation along this branch.')
                return
        
        # The bifurcation point is unique, do branch switching
        prev_index = -10 if len(branch.p_path) > 10 else 0 # point on the branch close enough to x_singular (with guarding for short branches)
        x_prev = np.append(branch.u_path[prev_index,:], branch.p_path[prev_index])
        directions, tangents = brs.branchSwitching(G, x_singular, x_prev, sp)

        # For each of the branches, run pseudo-arclength continuation
        M = len(u0)
        for n in range(len(directions)):
            x0 = directions[n]
            tangent = computeTangent(G, x0[0:M], x0[M], tangents[n], sp)
            _recursiveContinuation(G, x0[0:M], x0[M], tangent, ds_min, ds_max, ds, n_steps, sp, termination_event_index, result)