import numpy as np
import numpy.linalg as lg
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
			y = alpha*s
			for _ in range(1, m):
				s = s - alpha*A_mv(s)
				y = y + alpha*s
			return y
		return slg.LinearOperator((M+2,M+2), apply)
	B_inv = poly_inv(matvec, 1.0, min(M,10))
	y, _ = slg.lgmres(sys, rhs, x0=y_prev, M=B_inv, maxiter=10000)

	# Check if the l-gmres solver converged. If not, switch to a slow direct solver.
	#if y_prev is None or info > 0 or np.abs(y[M+1]) > 100:
	#	print('GMRES Failed, Switching to a Direct Solver with the full Jacobian.', y[M+1], lg.norm(matvec(y) - rhs))
	#	y = test_fn_bifurcation_exact(matvec, rhs)
	return y, y[M+1]

def test_fn_bifurcation_exact(matvec, rhs):
	# Construct the full matrix (yes, this is unfortunate but necessary...)
	A = np.zeros((rhs.size, rhs.size))
	for col  in range(rhs.size):
		A[:, col] = matvec(np.eye(rhs.size)[:, col])
	return lg.solve(A, rhs)

def test_fn_bifurcation_preconditioned(dF_w, x, l, r, M, y_prev, zr_prev, eps_reg=1e-5):
	# Build the bordered system and its right-hand side
	def matvec(w):
		el_1 = dF_w(x, w[0:M+1]) + eps_reg * w[0:M+1] + r*w[M+1]
		el_2 = np.dot(l, w[0:M+1])
		return np.append(el_1, el_2)
	sys = slg.LinearOperator((M+2, M+2), matvec)
	rhs = np.zeros(M+2); rhs[M+1] = 1.0

	# Build the operator A = nabla F
	def A(v):
		return dF_w(x, v) + eps_reg * v
	Asys = slg.LinearOperator((M+1, M+1), A)
	
	# Cache z_r = (A)^{-1} r once; also gamma = - l^T z_r
	print('zr_prev', zr_prev)
	z_r, _ = slg.lgmres(Asys, r, x0=zr_prev, maxiter=1000)
	gamma = - np.dot(l, z_r)

	# Define the preconditioner action Minv * rhs using the block inverse formula
	def Minv_mv(rhs):
        # rhs = [b; beta]
		b, beta = rhs[:M+1], float(rhs[M+1])
		# 1) y = A^{-1} b
		y_int, _ = slg.lgmres(Asys, b, maxiter=1000) # Need a good initial guess here
		# 2) scalar part
		s  = beta - np.dot(l, y_int)
		y_intp = s / (gamma + 1e-300)
		# 3) vector part
		y_intu = y_int - z_r * y_intp
		return np.append(y_intu, y_intp)
	Minv = slg.LinearOperator((M+2, M+2), Minv_mv)

	# Solve the original linear system but with a preconditioner.
	y, _ = slg.lgmres(sys, rhs, x0=y_prev, M=Minv, maxiter=1000)

	# Return
	return y, y[M+1], z_r



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

def is_bifurcation_point(Gu_v, Gp, u, p, tangent, M, z_vector, eps_reg=1e-5):
	rhs = Gp(u, p)

	# Solve with eps_reg
	J1v = lambda v: Gu_v(u, p, v) + eps_reg * v
	J1 = slg.LinearOperator((M, M), J1v)
	z1, info = slg.lgmres(J1, rhs, x0=z_vector, maxiter=1000)
	norm_1 = lg.norm(z1)

	# Solve with 10 * eps_reg
	J2v = lambda v: Gu_v(u, p, v) + 10.0 * eps_reg * v
	J2 = slg.LinearOperator((M, M), J2v)
	z2, info = slg.lgmres(J2, rhs, x0=z_vector, maxiter=1000)
	norm_2 = lg.norm(z2)

	# Solve with 100 * eps_reg
	J3v = lambda v: Gu_v(u, p, v) + 100.0 * eps_reg * v
	J3 = slg.LinearOperator((M, M), J3v)
	z3, info = slg.lgmres(J3, rhs, x0=z_vector, maxiter=1000)
	norm_3 = lg.norm(z3)

	print('Norms', norm_1, norm_2, norm_3)

def test_fn_extended_system(Gu_v, Gp, u, p, tangent, l, r, M, prev_z_vector, prev_zr_vector, eps_reg=1e-5):
	tu = tangent[0:M]
	tp = tangent[M]
	ru = r[0:M]
	rp = r[M]
	lu = l[0:M]
	lp = l[M]

	Jv = lambda v: Gu_v(u, p, v) + eps_reg * v
	J = slg.LinearOperator((M, M), Jv)

	# Do the first system solve for z
	rhs = Gp(u, p)
	z, _ = slg.lgmres(J, rhs, x0=prev_z_vector, maxiter=1000)
	zr, _ = slg.lgmres(J, ru, x0=prev_zr_vector, maxiter=1000)

	# Calculate s - same sign as det dF_w
	phi = tp - np.dot(tu, z)
	yp = (rp - np.dot(tu, zr)) / phi
	yu = zr - z * yp
	s = np.dot(lu, yu) + lp*yp
	print('phi, yp, yu, s', phi, yp, yu, s)

	# Return
	return s, z, zr
