import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt

from . import TestFunctions as tf

from typing import Callable, List, Tuple, Dict, Any

def computeTangent(u, p, Gu_v, Gp, prev_tangent, M, a_tol):
	"""
	This function computes the tangent to the curve at a given point by solving D_u G * tau + G_p = 0.
	The tangent vector then is [tau, 1] with normalization, and in the direction of prev_tangent.

	Parameters:
	----------
	u: ndarray
		The current state variable
	p: float 
		The current parameter value
	Gu_v: Callable
		The Matrix-free Jacobian of G(u,p) in the direction of v.
	Gp : callable
        Function calculating the derivative of G with respect to the parameter,
        with signature ``Gp(u, p) -> ndarray`` where `u` is the state vector and `p`
        is the continuation parameter.
	prev_tangent : ndarray
		The previous tangent vector along the curve (used for initial guess)
	M : int
		The size of the state variable
	a_tol: float
		The absolute tolerance for L-GMRES

	Returns
	"""

	DG = slg.LinearOperator((M, M), lambda v: Gu_v(u, p, v))
	b = -Gp(u, p)

	tau = slg.lgmres(DG, b, x0=prev_tangent[:M], atol=a_tol)[0]
	tangent = np.append(tau, 1.0)
	tangent = tangent / lg.norm(tangent)

	# Make sure the new tangent vector points in the same rough direction as the previous one
	if np.dot(tangent, prev_tangent) < 0.0:
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
    sp : dict
		Additional paramters for PyCont.

    Returns
    -------
    u_path: ndarray
		Two-dimensional array containining all state vectors along the branch in the first dimension.
		Size is (n, M) where n is the number of continuation points along the branch.
	p_path: ndarray
		One-dimensional array containining all parameter values along the branch.
	bifurcation_points: List
		Contains the bifurcation point, empty if none was detected.
    """    
	
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
	prev_tangent = initial_tangent / lg.norm(initial_tangent)

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
		N = lambda q: np.dot(tangent, q - x) - ds
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
	Localizes the bifurcation point between x_start and x_end using the bisection method.

    Parameters
	----------
        dF_w: Callable
			Function returning the Jacobian-vector product of the extended system, with signature
			``dF_w(x, w) -> ndarray`` where `x=(u,p)` is the full state vector and `w` is the 
			direction in which to compute the derivative
        x_start : ndarray 
			Starting point (u, p) to the 'left' of the bifurcation point.
        x_end : ndarray 
			End point (u, p) to the 'right' of the bifurcation point.
        l, r : ndarray
			Random vectors used during bifurcation detection.
        M : int
			Dimension of u.
        a_tol : float
			Absolute tolerance for Newton solver
        tau_vector_prev : ndarray
			Previous tau_vector in x_start used for bifurcation detection.
        max_bisect_steps : int
			Maximum allowed number of bisection steps.

    Returns
	-------
		is_bf : boolean
			True if there is an actual sign change in the test function, False for a fold point.
        x_bifurcation: ndarray (M+1,)
			The location of the bifurcation point within the tolerance a_tol.
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