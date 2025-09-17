import numpy as np
import scipy.sparse.linalg as slg

from typing import Callable, Tuple, Dict

def make_bordered_jacobian_system(F : Callable[[np.ndarray], np.ndarray], 
						 		  x0 : np.ndarray, 
						 		  l : np.ndarray, 
						 		  r : np.ndarray, 
						 		  rdiff : float, 
						 		  eps_reg : float) -> Callable[[np.ndarray], np.ndarray]:
	"""
	Create the matrix-free Jacobian of F.

	Returns
	-------
		matvec : Callable
			A function representing the directional derivative dF/dx(x0) * w.
	"""
	F0 = F(x0)
	Mp1 = F0.size

	def matvec(w):
		w_x = w[0:Mp1]
		norm_wx = np.linalg.norm(w_x)
		if norm_wx == 0.0:
			el1 = r*w[Mp1]
		else:
			el1 = (F(x0 + rdiff * w_x / norm_wx) - F(x0 - rdiff * w_x / norm_wx)) / (2.0 * rdiff / norm_wx) + r*w[Mp1]
		el2 = np.dot(l, w_x)
		return np.concatenate([el1, [el2]])

	return matvec

def solve_bordered_system_krylov(matvec : Callable[[np.ndarray], np.ndarray],
						         M : int, 
						         y_prev : np.ndarray | None) -> Tuple[np.ndarray, float]:
	sys = slg.LinearOperator((M+2, M+2), matvec)
	rhs = np.zeros(M+2); rhs[M+1] = 1.0

	maxiter = M+2 if M <= 10 else 20
	with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
		y, info = slg.lgmres(sys, rhs, x0=y_prev)
	residual = float(np.linalg.norm(matvec(y) - rhs))

	return y, residual

def test_fn_bifurcation(F : Callable[[np.ndarray], np.ndarray], 
						x : np.ndarray,
						l : np.ndarray, 
						r : np.ndarray, 
						M : int, 
						y_prev : np.ndarray | None, 
						sp : Dict,
						eps_reg : float =0) -> Tuple[np.ndarray, float, float]:
	"""
	Main test function to detect a bifurcation point. Bifurcation points are 
	locations x = (u, p) where Gu becomes singular and Gp lies in the column
	space of Gu. A bifurcation point is detected when the solution y to the bordered
	system [dFdx  r ; l^T 0] y = e_{M+2} switches sign in the last component y[M+1].

	Parameters
	----------
		F : Callable
			Extended objective F, of signature `F(x) -> ndarray`
			where `x=(u,p)` is the current point.
		x : ndarray
			Current point (u, p) on the branch.
		l : ndarray
			Left test vector for the bordered system.
		r : ndarray
			Right test vector for the bordered system.
		M : int
			Size of the state vector u.
		y_prev : ndarray
			Solution the bordered system at the previous point along the branch. Used
			as initial guess in the L-GMRES solver. Can be None.
		sp : Dict
			Solver parameters.
		eps_reg : float (default 1e-5)
			Regularization parameter when dFdx is ill-conditioned.

	Returns
	-------
		y : ndarray
			The solution to the bordered system
		phi : float
			Value of the test function, or y[M+1].

	Notes
	-----
		- This function implements the now-famous bifurcation detection algorithm from
		  [], specifically equation ().
		- A simple polynomial preconditioner of maximum order min(M, 10) is used to speed up
		  the L-GMRES solver for large-scale systems.
		- Spurious sign changes can happen at a fold point then the Jacobian Gu becomes
		  ill-conditioned and ||y|| explodes. Further checks must be done to classify 
		  fold points from real bifurcation points (implemented in PseudoArclengthContinuation.py).
		- Although this test function detects some fold points, it cannot be reliably
		  used to detect fold points in general
		- e_{M+2} is the M+2 - unit vector.
	"""

	
	matvec = make_bordered_jacobian_system(F, x, l, r, sp["rdiff"], eps_reg)
	y, residual = solve_bordered_system_krylov(matvec, M, y_prev)
	print('Test FN', y[M+1], residual)

	return y, y[M+1], residual