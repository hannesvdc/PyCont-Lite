import numpy as np
import scipy.sparse.linalg as slg

# Bifurcation Detection Test Function. We slightly regularize the system
# for better numerical convergence behavior in L-GMRES.
def test_fn_bifurcation(dF_w, x, l, r, M, y_prev, eps_reg=1.e-5):
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