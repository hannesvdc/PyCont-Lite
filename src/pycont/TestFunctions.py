import numpy as np
import scipy.sparse.linalg as slg

from typing import Callable, Tuple

# Bifurcation Detection Test Function. We slightly regularize the system
# for better numerical convergence behavior in L-GMRES.
def test_fn_bifurcation(dF_w : Callable[[np.ndarray, np.ndarray], np.ndarray], 
						x : np.ndarray,
						l : np.ndarray, 
						r : np.ndarray, 
						M : int, 
						y_prev : np.ndarray | None, 
						eps_reg : float =1.e-5) -> Tuple[np.ndarray, float]:
	"""
	Main test function to detect a bifurcation point. Bifurcation points are 
	locations x = (u, p) where Gu becomes singular and Gp lies in the column
	space of Gu. A bifurcation point is detected when the solution y to the bordered
	system [dF_w  r ; l^T 0] y = e_{M+2} switches sign in the last component y[M+1].

	Parameters
	----------
		dF_w : Callable
			Jacobian of the extended objective F, of signature `dF_w(x, w) -> ndarray`
			where `x=(u,p)` is the current point and `w` is a differentiation direction.
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
		eps_reg : float (default 1e-5)
			Regularization parameter when dF_w is ill-conditioned.

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

	def matvec(w):
		el_1 = dF_w(x, w[0:M+1]) + eps_reg * w[0:M+1] + r*w[M+1]
		el_2 = np.dot(l, w[0:M+1])
		return np.append(el_1, el_2)
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
	y, _ = slg.lgmres(sys, rhs, x0=y_prev, M=B_inv, maxiter=10000)

	return y, y[M+1]