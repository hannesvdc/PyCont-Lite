import numpy as np
import scipy.sparse.linalg as slg

from typing import Callable, Tuple, Dict

def make_bordered_jacobian_system(F : Callable[[np.ndarray], np.ndarray], 
						 x0 : np.ndarray, 
						 l : np.ndarray, 
						 r : np.ndarray, 
						 rdiff : float, 
						 eps_reg : float):
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
		el1 = (F(x0 + rdiff * w_x) - F0) / rdiff + eps_reg * w_x + r*w[Mp1]
		el2 = np.dot(l, w_x) + eps_reg * w[Mp1]
		return np.concatenate([el1, [el2]])

	return matvec

def solve_bordered_system_krylov(matvec : Callable[[np.ndarray], np.ndarray],
						         M : int, 
						        y_prev : np.ndarray | None) -> np.ndarray:
	sys = slg.LinearOperator((M+2, M+2), matvec)
	rhs = np.zeros(M+2); rhs[M+1] = 1.0

	# returns LinearOperator y ≈ A^{-1} b using p_m(A)=alpha*sum_{j=0}^{m-1}(I-alpha A)^j
	def poly_inv(A_mv, alpha, m):
		def apply(b):
			s = b.copy()
			vec = alpha*s
			for _ in range(1, m):
				s = s - alpha*A_mv(s)
				vec = vec + alpha*s
			return vec
		return slg.LinearOperator((M+2,M+2), apply)
	B_inv = poly_inv(matvec, 1.0, min(M,10))

	with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
		y, _ = slg.lgmres(sys, rhs, x0=y_prev, M=B_inv, maxiter=10000)

	return y

# Bifurcation Detection Test Function. We slightly regularize the system
# for better numerical convergence behavior in L-GMRES.
def test_fn_bifurcation(F : Callable[[np.ndarray], np.ndarray], 
						x : np.ndarray,
						l : np.ndarray, 
						r : np.ndarray, 
						M : int, 
						y_prev : np.ndarray | None, 
						sp : Dict,
						eps_reg : float =1.e-5) -> Tuple[np.ndarray, float]:
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
	y = solve_bordered_system_krylov(matvec, M, y_prev)

	return y, y[M+1]