import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt

from .Tangent import computeTangent
from .detection import DetectionModule

from .Types import Branch, Event
from .Logger import LOG

from typing import Callable, Tuple, Dict, Any, List

def continuation(G : Callable[[np.ndarray, float], np.ndarray], 
                 u0 : np.ndarray, 
                 p0 : float, 
                 initial_tangent : np.ndarray, 
                 ds_min : float, 
                 ds_max : float, 
                 ds : float, 
                 n_steps : int,
				 branch_id : int,
				 detectionModules : List[DetectionModule],
                 sp : Dict[str, Any],
				 high_accuracy : bool = True) -> Tuple[Branch, Event]:
	
	"""
    Function that performs the actual pseudo-arclength continuation of the current branch. It starts
	at the initial point (u0, p0), calculates the tangent along the curve, predicts the next points and
	corrects it using a matrix-free Newton-Krylov solver. At every iteration it checks for fold and
	bifurcation points.

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
    initial_tangent : ndarray
        Tangent to the current branch in (u0, p0)
    ds_min : float
        Minimum allowable continuation step size.
    ds_max : float
        Maximum allowable continuation step size.
    ds : float
        Initial continuation step size.
    n_steps : int
        Maximum number of continuation steps to perform.
    branch_id : int 
        Integer identifier of the current branch.
    detectionModules : List[DetectionModule]
        The list of active detection modules for this continuation stage.
    sp : dict
		Additional paramters for PyCont.

    Returns
    -------
	branch : Branch
		An instance of `Branch` that stores the complete branch and the reason it terminated, see the Branch dataclass
	event : Event
		An instance of `Event` that stores the reason why continuation terminated, as well as the location of the final
		point. Reasons include "BP" for a bifurcation point, "LP" for a fold, "MAXSTEPS" if we reached `n_steps` on the
		current branch, or "DSFLOOR" if the current arc length `ds` dips below `ds_min` and continuation failed due to this. 
    """    
	
	# Infer parameters from inputs
	M = len(u0)
	max_it = sp["nk_maxiter"]
	r_diff = sp["rdiff"]
	a_tol = sp["tolerance"]
	nk_tolerance = max(a_tol, r_diff) if high_accuracy else 1e-4

	# Initialize a point on the path
	x = np.append(u0, p0)
	s = 0.0
	tangent = initial_tangent / lg.norm(initial_tangent)
	branch = Branch(branch_id, n_steps, u0, p0)
	print_str = f"Step n: {0:3d}\t u: {lg.norm(u0):.4f}\t p: {p0:.4f}\t s: {s:.4f}\t t_p: {tangent[M]:.4f}"
	LOG.info(print_str)

	# Initialize all detection modules
	for module in detectionModules:
		module.initializeBranch(x, tangent)

	for n in range(1, n_steps+1):
		# Create the extended system for corrector
		N = lambda q: np.dot(tangent, q - x) - ds
		F = lambda q: np.append(G(q[0:M], q[M]), N(q))

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			
			# Predictor: Follow the tangent vector
			x_p = x + tangent * ds
			new_s = s + ds

			# Corrector
			with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
				try:
					x_new = opt.newton_krylov(F, x_p, f_tol=a_tol, rdiff=r_diff, maxiter=max_it, verbose=False)
				except opt.NoConvergence as e:
					x_new = e.args[0]
			nk_residual = lg.norm(F(x_new))
			
			# Check the residual to increase or decrease ds
			if np.all(np.isfinite(nk_residual)) and nk_residual <= 10.0 * nk_tolerance:
				ds = min(1.2*ds, ds_max)
				break
			else:
				ds = max(0.5*ds, ds_min)

		else:
			# This case should never happpen under normal circumstances
			LOG.info('Minimal Arclength Size is too large. Aborting.')
			termination_event = Event("DSFLOOR", x[0:M], x[M], s)
			branch.termination_event = termination_event
			return branch.trim(), termination_event
		
		# Determine the tangent to the curve at current point
		new_tangent = computeTangent(G, x_new[0:M], x_new[M], tangent, sp, high_accuracy=high_accuracy)
		
		# Go through all detection modules and check if we passed an important point
		if n % 5 == 0:
			for module in detectionModules:
				passed_point = module.update(F, x_new, new_tangent)
				if not passed_point:
					continue
				
				# Localize the special point and check if it is not None
				special_point = module.localize()
				if special_point is None: # Localization failed
					continue
				ds_special = float(np.linalg.norm(special_point - x))
				s_special = s + ds_special

				# Quit continuation and return to main driver.
				termination_event = Event(module.kind, special_point[0:M], special_point[M], s_special, info={"tangent" : np.copy(new_tangent)})
				termination_event = module.addTerminationInfo(termination_event)
				branch.addSpecialPoint(special_point, s_special)
				branch.termination_event = termination_event
				return branch.trim(), termination_event
		
		# Bookkeeping for the next step
		tangent = np.copy(new_tangent)
		x = np.copy(x_new)
		s = new_s
		branch.addPoint(x, s)
		
		# Print the status
		print_str = f"Step n: {n:3d}\t u: {lg.norm(x[0:M]):.4f}\t p: {x[M]:.4f}\t s: {s:.4f}\t t_p: {tangent[M]:.4f}"
		LOG.info(print_str)

	termination_event = Event("MAXSTEPS", branch.u_path[-1,:], branch.p_path[-1], branch.s_path[-1])
	branch.termination_event = termination_event
	return branch.trim(), termination_event