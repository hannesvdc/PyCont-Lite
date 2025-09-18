import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt

from .Tangent import computeTangent, computeFoldPoint
from .Bifurcation import computeBifurcationPoint, test_fn_jacobian, test_fn_bordered

from .Types import Branch, Event
from .Logger import LOG

from typing import Callable, Tuple, Dict, Any

def continuation(G : Callable[[np.ndarray, float], np.ndarray], 
                 u0 : np.ndarray, 
                 p0 : float, 
                 initial_tangent : np.ndarray, 
                 ds_min : float, 
                 ds_max : float, 
                 ds : float, 
                 n_steps : int,
				 branch_id : int,
                 sp : Dict[str, Any]) -> Tuple[Branch, Event]:
	
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
	bifurcation_detection = sp["bifurcation_detection"]
	param_min = sp["param_min"]
	param_max = sp["param_max"]
	nk_tolerance = max(a_tol, r_diff)

	# Initialize a point on the path
	x = np.append(u0, p0)
	s = 0.0
	tangent = initial_tangent / lg.norm(initial_tangent)
	branch = Branch(branch_id, n_steps, u0, p0)
	print_str = f"Step n: {0:3d}\t u: {lg.norm(u0):.4f}\t p: {p0:.4f}\t s: {s:.4f}\t t_p: {tangent[M]:.4f}"
	LOG.info(print_str)

	# Variables for test_fn bifurcation detection - Ensure no component in the direction of the tangent
	rng = rd.RandomState(seed=sp["seed"])
	r = rng.normal(0.0, 1.0, M+1); r = r / lg.norm(r)
	l = rng.normal(0.0, 1.0, M+1); l = l / lg.norm(l)
	r = r - np.dot(r, tangent) * tangent; r = r / lg.norm(r)
	l = l - np.dot(l, tangent) * tangent; l = l / lg.norm(l)
	prev_bf_w_vector = np.zeros(M+1)
	prev_bf_w_value = 0.0

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
			
			# Check the residual to increase or decrease ds
			nk_residual = lg.norm(F(x_new))
			good_residual = np.all(np.isfinite(nk_residual))
			if good_residual and nk_residual <= 10.0 * nk_tolerance:
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
		
		# Check that the new point does not exceed param_min or param_max, if supplied
		if param_min is not None and x_new[M] < param_min:
			LOG.info(f'Stopping Continuation Along this Branch. PARAM_MIN {param_min} reached.')
			termination_event = Event("PARAM_MIN", x_new[0:M], x_new[M], new_s)
			branch.termination_event = termination_event
			return branch.trim(), termination_event
		if param_max is not None and x_new[M] > param_max:
			LOG.info(f'Stopping Continuation Along this Branch. PARAM_MAX {param_max} reached.')
			termination_event = Event("PARAM_MAX", x_new[0:M], x_new[M], new_s)
			branch.termination_event = termination_event
			return branch.trim(), termination_event
		
		# Determine the tangent to the curve at current point
		new_tangent = computeTangent(G, x[0:M], x[M], tangent, sp)

		# Check whether we passed a fold point.
		if new_tangent[M] * tangent[M] < 0.0 and n > 5:
			is_fold_point, x_fold, alpha_fold = computeFoldPoint(G, x, x_new, new_tangent, ds, sp)
			if not is_fold_point:
				LOG.info('Erroneous Fold Point detection due to blow-up in tangent vector.')
			else:
				LOG.info(f'Fold point at {x_fold}')

				# Append the fold point and x_new to the current path
				s_fold = s + alpha_fold * (new_s - s)
				branch.addPoint(x_fold[0:M], x_fold[M], s_fold)
				branch.addPoint(x_new[0:M], x_new[M], new_s)
				
				# Stop continuation along this branch
				termination_event = Event("LP", x_fold[0:M], x_fold[M], s_fold, {"tangent": new_tangent})
				branch.termination_event = termination_event
				return branch.trim(), termination_event

		# Do bifurcation detection in the new point
		if bifurcation_detection and n % 5 == 0:
			bf_w_vector, bf_w_value = test_fn_jacobian(F, x_new, l, r, prev_bf_w_vector, sp)
			#bf_w_vector, bf_w_value = test_fn_bordered(F, x_new, l, r, prev_bf_w_vector, M, sp)

			if prev_bf_w_value * bf_w_value < 0.0 and (np.abs(bf_w_value) < 1000.0 or np.abs(prev_bf_w_value) < 1000.0): # Possible bifurcation point detected
				LOG.info(f'Sign change detected {prev_bf_w_value} {bf_w_value}')

				is_bf_point, x_singular, alpha_singular = computeBifurcationPoint(F, x, x_new, l, r, bf_w_vector, M, sp)
				if is_bf_point:
					LOG.info(f'Bifurcation Point at {x_singular}')
					s_singular = s + alpha_singular * (new_s - s)
					branch.addPoint(x_singular[0:M], x_singular[M], s_singular)
					termination_event = Event("BP", x_singular[0:M], x_singular[M], s_singular)
					branch.termination_event = termination_event
					return branch.trim(), termination_event
				else:
					LOG.info('Erroneous sign change in bifurcation detection, most likely due to blowup. Continuing along this branch.')
				
			prev_bf_w_value = bf_w_value
			prev_bf_w_vector = bf_w_vector

		# Bookkeeping for the next step
		tangent = np.copy(new_tangent)
		x = np.copy(x_new)
		s = new_s
		branch.addPoint(x[0:M], x[M], s)
		
		# Print the status
		print_str = f"Step n: {n:3d}\t u: {lg.norm(x[0:M]):.4f}\t p: {x[M]:.4f}\t s: {s:.4f}\t t_p: {tangent[M]:.4f}"
		LOG.info(print_str)

	termination_event = Event("MAXSTEPS", branch.u_path[-1,:], branch.p_path[-1], branch.s_path[-1])
	branch.termination_event = termination_event
	return branch.trim(), termination_event