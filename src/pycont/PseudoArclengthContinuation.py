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
	u = np.copy(u0) # Always the previous point on the curve
	p = np.copy(p0)	# Always the previous point on the curve
	u_path = [u]
	p_path = [p]

	print_str = 'Step n: {0:3d}\t u: {1:4f}\t p: {2:4f}'.format(0, lg.norm(u), p)
	print(print_str)

	# Choose intial tangent (guess). We need to negate to find the actual search direction
	prev_tangent = -initial_tangent / lg.norm(initial_tangent)

	# Variables for test_fn bifurcation detection - Ensure no component in the direction of the tangent
	#rng = rd.RandomState()
	#r = rng.normal(0.0, 1.0, M+1)
	#l = rng.normal(0.0, 1.0, M+1)
	#r = r - np.dot(r, prev_tangent) / np.dot(prev_tangent, prev_tangent) * prev_tangent
	#l = l - np.dot(l, prev_tangent) / np.dot(prev_tangent, prev_tangent) * prev_tangent
	#r = r / lg.norm(r)
	#l = l / lg.norm(l)
	#prev_tau_value = 0.0
	#prev_tau_vector = None

	# Improved bifurcation detection
	prev_phi_value = 0.0
	prev_z_vector = None

	for n in range(1, n_steps+1):
		# Determine the tangent to the curve at current point
		tangent = computeTangent(u, p, Gu_v, Gp, prev_tangent, M, a_tol)

		# Create the extended system for corrector
		N = lambda x: np.dot(tangent, x - np.append(u, p)) + ds
		F = lambda x: np.append(G(x[0:M], x[M]), N(x))
		dF_w = lambda x, w: (F(x + r_diff * w) - F(x)) / r_diff

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			# Predictor: Follow the tangent vector
			u_p = u + tangent[0:M] * ds
			p_p = p + tangent[M]   * ds
			x_p = np.append(u_p, p_p)

			# Corrector: Newton-Krylov
			try:
				x_result = opt.newton_krylov(F, x_p, f_tol=a_tol, rdiff=r_diff, maxiter=max_it, verbose=False)
				ds = min(1.2*ds, ds_max)
				break
			except:
				# Decrease arclength if the solver needs more than max_it iterations
				ds = max(0.5*ds, ds_min)
		else:
			# This case should never happpen under normal circumstances
			print('Minimal Arclength Size is too large. Aborting.')
			return np.array(u_path), np.array(p_path), []
		u_new = x_result[0:M]
		p_new = x_result[M]

		# Do bifurcation detection in the new point
		# if bifurcation_detection:
		# 	tau_vector, tau_value = tf.test_fn_bifurcation_fast(dF_w, np.append(u_new, p_new), l, r, M, prev_tau_vector)
		# 	if prev_tau_value * tau_value < 0.0: # Bifurcation point detected
		# 		print('Sign change detected', prev_tau_value, tau_value)
		# 		is_bf, x_singular = _computeBifurcationPointBisect(dF_w, np.append(u, p), np.append(u_new, p_new), l, r, M, a_tol, prev_tau_vector)
		# 		if is_bf:
		# 			return np.array(u_path), np.array(p_path), [x_singular]
				
		# 	prev_tau_value = tau_value
		# 	prev_tau_vector = tau_vector

		if bifurcation_detection:
			phi_value, z_vector = tf.test_fn_bifurcation_keller(Gu_v, Gp, u_new, p_new, tangent, M, prev_z_vector)
			print('z_vector', z_vector)
			if prev_phi_value * phi_value < 0.0:
				print('Sign change detected', prev_phi_value, phi_value)
				x_singular = _computeBifurcationPointNewton(G, Gu_v, u_new, p_new, z_vector, M, r_diff)

				# Classify fold point versus bifurcation point
				new_tangent = computeTangent(x_singular[0:M], x_singular[M], Gu_v, Gp, tangent, M, a_tol)
				print('Tangent at bifurcation', new_tangent, new_tangent[M])


			prev_phi_value = phi_value
			prev_z_vector = z_vector

		# Bookkeeping for the next step
		prev_tangent = np.copy(tangent)
		u = np.copy(u_new)
		p = np.copy(p_new)
		u_path.append(u)
		p_path.append(p)
		
		# Print the status
		print_str = 'Step n: {0:3d}\t u: {1:4f}\t p: {2:4f}'.format(n, lg.norm(u), p)
		print(print_str)

	return np.array(u_path), np.array(p_path), []

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
        x_bifurcation: array (M+1,), approximated bifurcation point
    """

	# Compute tau at start and end
	_, tau_start = tf.test_fn_bifurcation_fast(dF_w, x_start, l, r, M, tau_vector_prev)
	_, tau_end = tf.test_fn_bifurcation_fast(dF_w, x_end, l, r, M, tau_vector_prev)

	# Check that a sign change really exists
	if  tau_start * tau_end > 0.0:
		print("No sign change detected between start and end points.")
		return False, x_end

	for step in range(max_bisect_steps):
		x_mid = 0.5 * (x_start + x_end)
		_, tau_mid = tf.test_fn_bifurcation_fast(dF_w, x_mid, l, r, M, tau_vector_prev)

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

def _computeBifurcationPointNewton(G, Gu_v, u0, p0, z_vector, M, r_diff):
	# Setup the extended system
	def H(q : np.ndarray) -> np.ndarray:
		u_q = q[0:M]
		p_q = q[M]
		v_q = q[M+1:]

		crit_1 = G(u_q, p_q)
		crit_2 = Gu_v(u_q, p_q, v_q)
		crit_3 = 0.5 * (np.dot(v_q, v_q) - 1.0)
		return np.concatenate((crit_1, crit_2, [crit_3]))
	
	# Solve H = 0 using Newton-Krylov
	q0 = np.concatenate((u0, [p0], z_vector))
	q_bf = opt.newton_krylov(H, q0, rdiff=r_diff)
	print('Bifurcation Point found at', q_bf[0:M], q_bf[M])

	return q_bf[0:M+1]
