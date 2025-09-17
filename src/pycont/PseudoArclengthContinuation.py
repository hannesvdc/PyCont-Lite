import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt

from .Tangent import computeTangent
from .Bifurcation import computeBifurcationPointH, test_fn_jacobian

from .Types import Branch, Event

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
	tangent = computeTangent(G, u0, p0, initial_tangent / lg.norm(initial_tangent), sp)
	branch = Branch(branch_id, n_steps, u0, p0)
	print_str = f"Step n: {0:3d}\t u: {lg.norm(u0):.4f}\t p: {p0:.4f}\t s: {s:.4f}\t t_p: {tangent[M]:.4f}"
	print(print_str)

	# Variables for test_fn bifurcation detection - Ensure no component in the direction of the tangent
	rng = rd.RandomState()
	r = rng.normal(0.0, 1.0, M+1); r = r / lg.norm(r)
	l = rng.normal(0.0, 1.0, M+1); l = l / lg.norm(l)
	r = r - np.dot(r, tangent) * tangent; r = r / lg.norm(r)
	l = l - np.dot(l, tangent) * tangent; l = l / lg.norm(l)
	prev_bf_w_vector = None
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
			print('Minimal Arclength Size is too large. Aborting.')
			termination_event = Event("DSFLOOR", x[0:M], x[M], s)
			branch.termination_event = termination_event
			return branch.trim(), termination_event
		
		# Check that the new point does not exceed param_min or param_max, if supplied
		if param_min is not None and x_new[M] < param_min:
			termination_event = Event("PARAM_MIN", x_new[0:M], x_new[M], new_s)
			branch.termination_event = termination_event
			return branch.trim(), termination_event
		if param_max is not None and x_new[M] > param_max:
			termination_event = Event("PARAM_MAX", x_new[0:M], x_new[M], new_s)
			branch.termination_event = termination_event
			return branch.trim(), termination_event
		
		# Determine the tangent to the curve at current point
		new_tangent = computeTangent(G, x[0:M], x[M], tangent, sp)

		# Check whether we passed a fold point.
		if new_tangent[M] * tangent[M] < 0.0 and n > 5:
			#is_fold, x_fold, alpha_fold = _computeFoldPointBisect(G, x, x_new, tangent[M], new_tangent[M], tangent, ds, sp)
			is_fold_point, x_fold, alpha_fold = _computeFoldPoint(G, x, x_new, new_tangent, ds, sp)
			if not is_fold_point:
				print('Erroneous Fold Point detection due to blow-up in tangent vector.')
			else:
				print('Fold point at', x_fold)

				# Append the fold point and x_new to the current path
				s_fold = s + alpha_fold * (new_s - s)
				branch.addPoint(x_fold[0:M], x_fold[M], s_fold)
				branch.addPoint(x_new[0:M], x_new[M], new_s)
				
				# Stop continuation along this branch
				termination_event = Event("LP", x_fold[0:M], x_fold[M], s_fold, {"tangent": new_tangent})
				branch.termination_event = termination_event
				return branch.trim(), termination_event

		# Do bifurcation detection in the new point
		if bifurcation_detection:
			bf_w_vector, bf_w_value = test_fn_jacobian(F, x_new, l, r, M, prev_bf_w_vector, sp)

			if prev_bf_w_value * bf_w_value < 0.0 and np.abs(bf_w_value) < 1000.0: # Possible bifurcation point detected
				print('Sign change detected', prev_bf_w_vector, bf_w_value)

				is_bf_point, x_singular, alpha_singular = computeBifurcationPointH(F, x, x_new, l, r, bf_w_vector, M, sp)
				if is_bf_point:
					print('Bifurcation Point at', x_singular)
					s_singular = s + alpha_singular * (new_s - s)
					branch.addPoint(x_singular[0:M], x_singular[M], s_singular)
					termination_event = Event("BP", x_singular[0:M], x_singular[M], s_singular)
					branch.termination_event = termination_event
					return branch.trim(), termination_event
				else:
					print('Erroneous sign change in bifurcation detection, most likely due to blowup. Continuing along this branch.')
				
			prev_bf_w_value = bf_w_value
			prev_bf_w_vector = bf_w_vector

		# Bookkeeping for the next step
		tangent = np.copy(new_tangent)
		x = np.copy(x_new)
		s = new_s
		branch.addPoint(x[0:M], x[M], s)
		
		# Print the status
		print_str = f"Step n: {n:3d}\t u: {lg.norm(x[0:M]):.4f}\t p: {x[M]:.4f}\t s: {s:.4f}\t t_p: {tangent[M]:.4f}"
		print(print_str)

	termination_event = Event("MAXSTEPS", branch.u_path[-1,:], branch.p_path[-1], branch.s_path[-1])
	branch.termination_event = termination_event
	return branch.trim(), termination_event

# def _computeBifurcationPointBisect(F : Callable[[np.ndarray], np.ndarray], 
# 								   x_start : np.ndarray, 
# 								   x_end : np.ndarray, 
# 								   l : np.ndarray, 
# 								   r : np.ndarray, 
# 								   tau_vector_prev : Optional[np.ndarray],
# 								   sp : Dict,
# 								   max_bisect_steps : int=30) -> Tuple[bool, np.ndarray, float]:
# 	"""
# 	Localizes the bifurcation point between x_start and x_end using the bisection method.

#     Parameters
# 	----------
#         F: Callable
# 			Extended objective function with signature ``F(x) -> ndarray`` where `x=(u,p)` is the full state vector.
#         x_start : ndarray 
# 			Starting point (u, p) to the 'left' of the bifurcation point.
#         x_end : ndarray 
# 			End point (u, p) to the 'right' of the bifurcation point.
#         l, r : ndarray
# 			Random vectors used during bifurcation detection.
#         tau_vector_prev : ndarray
# 			Previous tau_vector in x_start used for bifurcation detection, can be None.
# 		sp : Dict
# 			Solver parameters.
#         max_bisect_steps : int
# 			Maximum allowed number of bisection steps.

#     Returns
# 	-------
# 		is_bf : boolean
# 			True if there is an actual sign change in the test function, False for a fold point.
#         x_bifurcation: ndarray (M+1,)
# 			The location of the bifurcation point within the tolerance a_tol.
#     """
# 	a_tol = sp["tolerance"]
# 	M = len(x_start) - 1

# 	# Compute tau at start and end
# 	_, tau_start, _ = test_fn_jacobian(F, x_start, l, r, M, tau_vector_prev, sp)
# 	_, tau_end, _ = test_fn_jacobian(F, x_end, l, r, M, tau_vector_prev, sp)

# 	# Check that a sign change really exists
# 	if  tau_start * tau_end > 0.0:
# 		print("No sign change detected between start and end points.")
# 		return False, x_end, -1.0

# 	alpha_start = 0.0
# 	alpha_end = 1.0
# 	for _ in range(max_bisect_steps):
# 		x_mid = 0.5 * (x_start + x_end)
# 		alpha = 0.5 * (alpha_start + alpha_end)
# 		_, tau_mid, _ = test_fn_jacobian(F, x_mid, l, r, M, tau_vector_prev, sp)

# 		# Narrow the interval based on sign of tau
# 		if tau_start * tau_mid < 0.0:
# 			x_end = x_mid
# 			alpha_end = alpha
# 			tau_end = tau_mid
# 		else:
# 			x_start = x_mid
# 			alpha_start = alpha
# 			tau_start = tau_mid

# 		# Convergence check
# 		if np.abs(tau_mid) < a_tol:
# 			print('Bisection converged', tau_mid)
# 			return True, 0.5 * (x_start + x_end), 0.5 * (alpha_start + alpha_end)

# 	print('Warning: Bisection reached maximum steps without full convergence.')
# 	x_mid = 0.5 * (x_start + x_end)
# 	alpha = 0.5 * (alpha_start + alpha_end)
# 	return True, x_mid, alpha # np.abs(tau_mid) < 1.0

def _computeFoldPoint(G : Callable[[np.ndarray, float], np.ndarray],
					  x_left : np.ndarray,
					  x_right : np.ndarray,
					  tangent_ref : np.ndarray,
					  ds : float,
					  sp : Dict) -> Tuple[bool, np.ndarray, float]:
	rdiff = sp["rdiff"]

	def make_F_ext(alpha : float) -> Callable[[np.ndarray], np.ndarray]:
		ds_alpha = alpha * ds
		N = lambda q: np.dot(tangent_ref, q - x_left) - ds_alpha
		F = lambda q: np.append(G(q[0:-1], q[-1]), N(q))
		return F
	def finalTangentComponent(alpha):
		F = make_F_ext(alpha)
		with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
			x_alpha = opt.newton_krylov(F, x_left, rdiff=rdiff)
		tangent = computeTangent(G, x_alpha[0:-1], x_alpha[-1], tangent_ref, sp)
		return tangent[-1]
	
	try:
		alpha_fold, result = opt.brentq(finalTangentComponent, -2.0, 2.0, full_output=True, disp=False)
	except ValueError: # No sign change detected
		return False, x_right, 1.0
	except opt.NoConvergence:
		return False, x_left, 1.0
	
	x_fold = x_left + alpha_fold * (x_right - x_left)
	return True, x_fold, alpha_fold
	

def _computeFoldPointBisect(G : Callable[[np.ndarray, float], np.ndarray],
							x_left : np.ndarray,
							x_right : np.ndarray,
							value_left : float,
							value_right : float,
							tangent_ref : np.ndarray,
							ds : float,
							sp : Dict,
							max_bisect_steps : int=20) -> Tuple[bool, np.ndarray, float]:
	"""
	Localizes the fold point between x_left and x_right using the bisection method.

    Parameters
	----------
        G : callable
			Function representing the nonlinear system, with signature
			``G(u, p) -> ndarray`` where `u` is the state vector and `p`
			is the continuation parameter.
        x_left : ndarray 
			Starting point (u, p) to the 'left' of the bifurcation point.
        x_right : ndarray 
			End point (u, p) to the 'right' of the bifurcation point.
        value_left : float
			Tangent value at x_left.
		value_right : float
			Tangent value at x_right.
		tangent_ref : np.ndarray
			Reference tangent (typically at x_left) to speed up tangent computations.
		ds : float
			Total arclength between x_left and x_right.
        sp : Dict
			Solver parameters.
        max_bisect_steps : int
			Maximum allowed number of bisection steps.

    Returns
	-------
		is_fold : bool
			True if the bisection algorithm found a fold point, False otherwise.
        x_fold: ndarray
			The location of the fold point within the tolerance a_tol.
    """
	a_tol = sp["tolerance"]
	rdiff = sp["rdiff"]

	if value_left * value_right > 0.0:
		print('Left and Right value have the same sign. Bisection will not work. Returning')
		return False, x_left, 0.0
	
	def make_F_ext(alpha : float) -> Callable[[np.ndarray], np.ndarray]:
		ds_alpha = alpha * ds
		N = lambda q: np.dot(tangent_ref, q - x_left) - ds_alpha
		F = lambda q: np.append(G(q[0:-1], q[-1]), N(q))
		return F
	def finalTangentComponent(alpha):
		F = make_F_ext(alpha)
		with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
			x_alpha = opt.newton_krylov(F, x_left, rdiff=rdiff)
		tangent = computeTangent(G, x_alpha[0:-1], x_alpha[-1], tangent_ref, sp)
		return tangent[-1], x_alpha
	
	alpha_left, alpha_right = 0.0, 1.0
	for _ in range(max_bisect_steps):
		alpha = 0.5 * (alpha_left + alpha_right)
		value, x_alpha = finalTangentComponent(alpha)

		if value * value_left < 0.0:
			alpha_right = alpha
			value_right = value
			x_right = x_alpha
		else:
			alpha_left = alpha
			value_left = value
			x_left = x_alpha

		# Convergence check
		if np.abs(value) < a_tol:
			return True, 0.5 * (x_left + x_right), 0.5 * (alpha_left + alpha_right)
		
	print('Warning: Bisection reached maximum steps without full convergence.')
	return np.abs(value) < 0.1, 0.5 * (x_left + x_right), 0.5 * (alpha_left + alpha_right)