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
	y, info = slg.lgmres(sys, rhs, x0=y_prev, maxiter=10000)

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

def test_fn_bifurcation_fast(dF_w, x, l, r, M, y_prev, eps_reg=1e-6):
	"""
    Fast/stable Beyn-Keller test function via Schur complement.

    Computes phi = -l^T (J+eps I)^{-1} r using one solve of size (M+1)x(M+1).
    Returns y, phi
    """
	
	# Build (J + eps I) v using the Jacobian-vector product dF_w at point x
	def Jv(v):
		return dF_w(x, v) + eps_reg * v
	sys = slg.LinearOperator((M+1, M+1), Jv)
	y, info = slg.lgmres(sys, r, x0=y_prev, maxiter=1000)
	phi = -np.dot(l, y)

	# Check if the l-gmres solver converged. If not, switch to a slow direct solver.
	gmres_residual = lg.norm(Jv(y) - r)
	if gmres_residual > 1000 * eps_reg:
		print('GMRES Failed, Switching to a Direct Solver with the full Jacobian.', phi, gmres_residual)
		y = test_fn_bifurcation_exact(Jv, r)
		phi = -np.dot(l, y)
		print('New Solution with Full Jacobian', phi, lg.norm(Jv(y) - r))

	return y, phi

def test_fn_bifurcation_keller(Gu_v, Gp, u, p, tangent, M, z_prev, eps_reg=1e-5):
	t_u = tangent[0:M]
	t_p = tangent[M]

	# Solve the linear system for z = (Gu(u, p) + eps I)^{-1} Gp
	Jv = lambda v: Gu_v(u, p, v) + eps_reg * v
	J = slg.LinearOperator((M, M), Jv)
	rhs = Gp(u, p)
	z, info = slg.lgmres(J, rhs, x0=z_prev, maxiter=1000)
	gmres_residual = lg.norm(Jv(z) - rhs)
	phi = t_p - np.dot(t_u, z)
	print('gmres residual', gmres_residual, phi)

	return phi, z