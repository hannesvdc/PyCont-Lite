import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt

from . import TestFunctions as tf

from typing import Callable, List, Tuple, Dict, Any

def computeTangent(u, p, Gu_v, Gp, prev_tau, M, a_tol):
	"""
	This function computes the tangent to the curve at a given point by solving D_u G * tau = - G_p.
	The tangent vector then is [tau, 1] with normalization.

	The arguments are:
		- u: The current state variable
		- p: The current parameter
		- Gu_v: The Jacobian of the system with respect to the state variable as a function of u, p, v
		- Gp: The Jacobian of the system with respect to the parameter as a function of u and p
		- prev_tau: The previous tangent vector (used for initial guess)
		- M: The size of the state variable
		- a_tol: The absolute tolerance for the Newton-Raphson solver
	"""

	DG = slg.LinearOperator((M, M), lambda v: Gu_v(u, p, v))
	b = -Gp(u, p)

	tau = slg.lgmres(DG, b, x0=prev_tau[:M], atol=a_tol)[0]
	tangent = np.append(tau, 1.0)
	tangent = tangent / lg.norm(tangent)

	# Make sure the new tangent vector points in the same rough direction as the previous one
	if np.dot(tangent, prev_tau) < 0.0:
		tangent = -tangent
	return tangent

def continuation(G : Callable[[np.ndarray, float], np.ndarray], 
                 Gu_v : Callable[[np.ndarray, float, np.ndarray], np.ndarray], 
                 Gp : Callable[[np.ndarray, float], np.ndarray], 
                 u0 : np.ndarray, 
                 p0 : float, 
                 initial_tangent : np.ndarray, 
                 ds_min : float, 
                 ds_max : float, 
                 ds : float, 
                 n_steps : int, 
                 sp : Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List]:
	
	# Infer parameters from inputs
	M = u0.size
	a_tol = sp["tolerance"]
	max_it = sp["nk_maxiter"]
	r_diff = sp["rdiff"]
	bifurcation_detection = sp["bifurcation_detection"]

	# Initialize a point on the path
	x = np.append(u0, p0)
	u_path = np.zeros((n_steps+1, M)); u_path[0,:] = u0
	p_path = np.zeros(n_steps+1); p_path[0] = p0

	# Choose intial tangent (guess). We need to negate to find the actual search direction
	prev_tangent = -initial_tangent / lg.norm(initial_tangent)

	print_str = f"Step n: {0:3d}\t u: {lg.norm(u0):.4f}\t p: {p0:.4f}\t t_p: {prev_tangent[M]:.4f}"
	print(print_str)

	# Variables for test_fn bifurcation detection - Ensure no component in the direction of the tangent
	rng = rd.RandomState()
	r = rng.normal(0.0, 1.0, M+1)
	l = rng.normal(0.0, 1.0, M+1)
	r = r - np.dot(r, prev_tangent) * prev_tangent; r = r / lg.norm(r)
	l = l - np.dot(l, prev_tangent) * prev_tangent; l = l / lg.norm(l)
	prev_tau_value = 0.0
	prev_tau_vector = None

	for n in range(1, n_steps+1):
		# Determine the tangent to the curve at current point
		tangent = computeTangent(x[0:M], x[M], Gu_v, Gp, prev_tangent, M, a_tol)

		# Create the extended system for corrector
		N = lambda q: np.dot(tangent, q - x) + ds
		F = lambda q: np.append(G(q[0:M], q[M]), N(q))
		dF_w = lambda q, w: (F(q + r_diff * w) - F(q)) / r_diff

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			# Predictor: Follow the tangent vector
			x_p = x + tangent * ds

			# Corrector: Newton-Krylov
			try:
				x_new = opt.newton_krylov(F, x_p, f_tol=a_tol, rdiff=r_diff, maxiter=max_it, verbose=False)
				ds = min(1.2*ds, ds_max)
				break
			except:
				# Decrease arclength if the solver needs more than max_it iterations
				ds = max(0.5*ds, ds_min)
		else:
			# This case should never happpen under normal circumstances
			print('Minimal Arclength Size is too large. Aborting.')
			return u_path[0:n,:], p_path[0:n], []

		# Do a simple fold detection
		if tangent[M] * prev_tangent[M] < 0.0 and n > 1: # Do not check in the first point
			print('Fold point near', x_new)

		# Do bifurcation detection in the new point
		if bifurcation_detection:
			tau_vector, tau_value = tf.test_fn_bifurcation(dF_w, x_new, l, r, M, prev_tau_vector)
			if prev_tau_value * tau_value < 0.0: # Bifurcation point detected
				print('Sign change detected', prev_tau_value, tau_value)

				is_bf, x_singular = _computeBifurcationPointBisect(dF_w, x, x_new, l, r, M, a_tol, prev_tau_vector)
				if is_bf:
					print('Bifurcation Point at', x_singular)
					return u_path[0:n,:], p_path[0:n], [x_singular]
			prev_tau_value = tau_value
			prev_tau_vector = tau_vector

		# Bookkeeping for the next step
		prev_tangent = np.copy(tangent)
		x = np.copy(x_new)
		u_path[n,:] = x[0:M]
		p_path[n] = x[M]
		
		# Print the status
		print_str = f"Step n: {n:3d}\t u: {lg.norm(x[0:M]):.4f}\t p: {x[M]:.4f}\t t_p: {tangent[M]:.4f}"
		print(print_str)

	return u_path, p_path, []

def _computeBifurcationPointBisect(dF_w, x_start, x_end, l, r, M, a_tol, tau_vector_prev, max_bisect_steps=30):
	"""
	Localizes the bifurcation point between x_start and x_end using bisection.

    Parameters:
        dF_w: function for Jacobian-vector product
        x_start: array (M+1,), start point [u, p]
        x_end: array (M+1,), end point [u, p]
        l, r: random bifurcation detection vectors (fixed)
        M: dimension of u
        a_tol: absolute tolerance for Newton solver
        tau_vector_prev: previous tau_vector (can be None)
        max_bisect_steps: maximum allowed bisection steps

    Returns:
		is_bf: boolean, True if there is an actual sign change in the test 
		       function, False for a fold pint
        x_bifurcation: array (M+1,), approximated bifurcation point
    """

	# Compute tau at start and end
	_, tau_start = tf.test_fn_bifurcation(dF_w, x_start, l, r, M, tau_vector_prev)
	_, tau_end = tf.test_fn_bifurcation(dF_w, x_end, l, r, M, tau_vector_prev)

	# Check that a sign change really exists
	if  tau_start * tau_end > 0.0:
		print("No sign change detected between start and end points.")
		return False, x_end

	for step in range(max_bisect_steps):
		x_mid = 0.5 * (x_start + x_end)
		_, tau_mid = tf.test_fn_bifurcation(dF_w, x_mid, l, r, M, tau_vector_prev)

		# Narrow the interval based on sign of tau
		if tau_start * tau_mid < 0.0:
			x_end = x_mid
			tau_end = tau_mid
		else:
			x_start = x_mid
			tau_start = tau_mid

		# Convergence check
		if np.linalg.norm(x_end - x_start) < a_tol:
			return True, 0.5 * (x_start + x_end)

	print('Warning: Bisection reached maximum steps without full convergence.')
	return True, 0.5 * (x_start + x_end)