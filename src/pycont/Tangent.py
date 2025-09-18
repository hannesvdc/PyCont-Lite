import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt

from .Logger import LOG

from typing import Callable, Dict, Tuple

def computeTangent(G: Callable[[np.ndarray, float], np.ndarray],
				   u : np.ndarray, 
				   p : float, 
				   prev_tangent : np.ndarray, 
				   sp : Dict) -> np.ndarray:
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
    if tangent_residual > 0.01:
        # Solve the linear system using Newton-Krylov with much better lgmres arguments
        def F(v):
            return matvec(v) - rhs
        tangent = opt.newton_krylov(F, prev_tangent, rdiff=rdiff, verbose=False)
        tangent_residual = lg.norm(F(tangent))
        LOG.verbose(f'Tangent Newton-Krylov Residual {tangent_residual}')

    # Make sure the new tangent lies in the direction of the previous one and return
    tangent = np.sign(np.dot(tangent, prev_tangent)) * tangent / lg.norm(tangent)
    return tangent

def computeFoldPoint(G : Callable[[np.ndarray, float], np.ndarray],
					  x_left : np.ndarray,
					  x_right : np.ndarray,
					  tangent_ref : np.ndarray,
					  ds : float,
					  sp : Dict) -> Tuple[bool, np.ndarray, float]:
	"""
	Localizes the bifurcation point between x_start and x_end using the bisection method.

    Parameters
	----------
        G: Callable
			Objective function with signature ``G(u,p) -> ndarray``
        x_left : ndarray 
			Starting point (u, p) to the 'left' of the fold point.
        x_right : ndarray 
			End point (u, p) to the 'right' of the fold point.
        tangent_ref : ndarray
			A reference tangent vector to speed up tangent calculations. Typically the 
			tangent vector at x_left.
		sp : Dict
			Solver parameters.

    Returns
	-------
		is_fold_point : boolean
			True if we detected an antual fold point.
        x_fold: ndarray
			The location of the fold point within the tolerance.
	"""
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