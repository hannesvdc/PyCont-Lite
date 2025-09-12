import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg

from typing import Callable, Dict

def _newton_schur(G: Callable[[np.ndarray, float], np.ndarray],
				  u : np.ndarray, 
				  p : float, 
				  prev_tangent : np.ndarray, 
				  current_guess : np.ndarray,
				  G_value : np.ndarray,
				  Gp : np.ndarray,
				  M : int,
				  rdiff : float,
				  tolerance : float):
    print('Trying one iteration of Newton-Schur to improve tangent condition')
    tp = current_guess[M]
    cu = prev_tangent[0:M]
    cp = prev_tangent[M]
	
    # Solve for t_u
    Gu_matvec = lambda v: (G(u + rdiff * v, p) - G_value) / rdiff
    J = slg.LinearOperator((M,M), Gu_matvec)
    rhs = -Gp * tp
    tu, _ = slg.lgmres(J, rhs, x0=current_guess[0:M], maxiter=min(M+2, 20), atol=tolerance, rtol=1e-6)
	
    # Evaluate phi and its derivative
    phi = np.dot(cu, tu) + cp * tp - 1.0
    if np.abs(tp) > 0.01:
        y = tu / tp
    else:
        y, _ = slg.lgmres(J, -Gp, x0=tu/tp, maxiter=min(M+2, 20), atol=tolerance, rtol=1e-6)
    phi_prime = -np.dot(cu, y) + cp
	
    # Update the guess for tp and tu
    tp = tp - phi / phi_prime
    tangent = np.append(tu, tp)
	
    # Return the normalized tangent
    return np.sign(np.dot(tangent, prev_tangent)) * tangent / lg.norm(tangent)

def computeTangent(G: Callable[[np.ndarray, float], np.ndarray],
				   u : np.ndarray, 
				   p : float, 
				   prev_tangent : np.ndarray, 
				   sp : Dict, 
				   eps_reg=1e-5) -> np.ndarray:
    rdiff = sp["rdiff"]
    tolerance = sp["tolerance"]
    atol = max(tolerance, rdiff)
    M = len(u)

    # Create the linear system and right-hand side
    G_value = G(u, p)
    Gp = (G(u, p + rdiff) - G_value) / rdiff
    def matvec(v):
        J = (G(u + rdiff * v[0:M], p) - G_value) / rdiff + eps_reg * v[0:M] + Gp * v[M]
        eq_2 = np.dot(prev_tangent, v) + eps_reg * v[M]
        return np.append(J, eq_2)
    sys = slg.LinearOperator((M+1, M+1), matvec)
    rhs = np.zeros(M+1); rhs[M] = 1.0

	# Solve the linear system and do postprocessing
    tangent, info = slg.lgmres(sys, rhs, x0=prev_tangent, maxiter=min(M+2, 10), atol=atol)
    tangent = np.sign(np.dot(tangent, prev_tangent)) * tangent / lg.norm(tangent)
    return tangent