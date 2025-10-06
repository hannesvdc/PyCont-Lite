import numpy as np
import scipy.optimize as opt

from ..Tangent import computeTangent
from ..Logger import LOG

from typing import Callable, Dict, Tuple

def computeFoldPoint(G : Callable[[np.ndarray, float], np.ndarray],
					 x_left : np.ndarray,
					 x_right : np.ndarray,
					 tangent_ref : np.ndarray,
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
        x_fold : ndarray
			The location of the fold point within the tolerance.
		alpha_fold : float
			Location of `x_fold` as a fraction between `x_left` and `x_right`.
	"""
	rdiff = sp["rdiff"]
	M = len(x_left)-1
	ds = np.linalg.norm(x_right - x_left)

	def make_F_ext(alpha : float) -> Callable[[np.ndarray], np.ndarray]:
		ds_alpha = alpha * ds
		N = lambda q: np.dot(tangent_ref, q - x_left) - ds_alpha
		F = lambda q: np.append(G(q[0:M], q[M]), N(q))
		return F
	def finalTangentComponent(alpha):
		F = make_F_ext(alpha)
		with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
			x_alpha = opt.newton_krylov(F, x_left, rdiff=rdiff)
		tangent = computeTangent(G, x_alpha[0:M], x_alpha[M], tangent_ref, sp)
		return tangent[M]
	
	try:
		LOG.verbose(f'BrentQ edge values {finalTangentComponent(-1.0)},  {finalTangentComponent(2.0)}')
		alpha_fold, result = opt.brentq(finalTangentComponent, -2.0, 2.0, full_output=True, disp=False)
	except ValueError: # No sign change detected
		return False, x_right, 1.0
	except opt.NoConvergence:
		return False, x_right, 1.0
	
	x_fold = x_left + alpha_fold * (x_right - x_left)
	return True, x_fold, alpha_fold