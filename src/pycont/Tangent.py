import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt

from typing import Callable, Dict

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
    if tangent_residual > 0.01:
        # Solve the linear system using Newton-Krylov with much better lgmres arguments
        def F(v):
            return matvec(v) - rhs
        tangent = opt.newton_krylov(F, prev_tangent, rdiff=rdiff, verbose=False)

    # Make sure the new tangent lies in the direction of the previous one and return
    tangent = np.sign(np.dot(tangent, prev_tangent)) * tangent / lg.norm(tangent)
    return tangent