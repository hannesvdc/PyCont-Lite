import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg

# Bifurcation Detection Test Function. We slightly regularize the system
# for better numerical convergence behavior in L-GMRES.
def test_fn_bifurcation(dF_w, x, l, r, M, y_prev, eps_reg=1.e-6):
	def matvec(w):
		el_1 = dF_w(x, w[0:M+1]) + eps_reg * w[0:M+1] + r*w[M+1]
		el_2 = np.dot(l, w[0:M+1])
		return np.append(el_1, el_2)
	sys = slg.LinearOperator((M+2, M+2), matvec)
	rhs = np.zeros(M+2); rhs[M+1] = 1.0
	y, info = slg.gmres(sys, rhs, x0=y_prev, maxiter=10000)

	# Check if the l-gmres solver converged. If not, switch to a slow direct solver.
	if y_prev is None or info > 0 or np.abs(y[M+1]) > 100:
		print('GMRES Failed, Switching to a Direct Solver with the full Jacobian.', y[M+1], lg.norm(matvec(y) - rhs))
		y = test_fn_bifurcation_exact(matvec, rhs)
	return y, y[M+1]

def test_fn_bifurcation_exact(matvec, rhs):
	# Construct the full matrix (yes, this is unfortunate but necessary...)
	A = np.zeros((rhs.size, rhs.size))
	for col  in range(rhs.size):
		A[:, col] = matvec(np.eye(rhs.size)[:, col])
	return lg.solve(A, rhs)