import numpy as np
import scipy.optimize as opt
import scipy.sparse.linalg as slg

from typing import Callable, Tuple, Dict

def test_fn_jacobian(F : Callable[[np.ndarray], np.ndarray], 
					 x : np.ndarray,
					 l : np.ndarray, 
					 r : np.ndarray, 
					 M : int, 
					 w_prev : np.ndarray | None, 
					 sp : Dict) -> Tuple[np.ndarray, float]:
    rdiff = sp["rdiff"]
    maxiter = M if M < 10 else 10
    def matvec(w):
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return 0.0 * w
        eps = rdiff / norm_w
        return (F(x + eps * w) - F(x - eps * w)) / (2.0*eps) # Exactly linear central differences
    sys = slg.LinearOperator((M+1, M+1), matvec)
      
    # Solve the linear system using L-GMRES. If the residual is too large, refine with Newton-Krylov.
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        w_solution, info = slg.lgmres(sys, r, x0=w_prev, maxiter=maxiter)
    residual = float(np.linalg.norm(matvec(w_solution) - r))
    if residual > 0.01:
        F_NK = lambda w: matvec(w) - r
        w_solution = opt.newton_krylov(F_NK, w_solution, rdiff=rdiff, verbose=False)
        residual = np.linalg.norm(F_NK(w_solution))
    beta = -1.0 / np.dot(l, w_solution)
    print('Test FN', beta, residual, info)

    return w_solution, beta

def computeBifurcationPoint(F : Callable[[np.ndarray], np.ndarray], 
							x_start : np.ndarray,
                            x_end : np.ndarray,
							l : np.ndarray, 
							r : np.ndarray,
                            w : np.ndarray,
                            M : int,
							sp : Dict) -> Tuple[bool, np.ndarray, float]:
    rdiff = sp["rdiff"]
    x_diff = x_end - x_start
    S = np.dot(l, w)
    z0 = np.append(w / S, -1.0 / S)

    # Build the Bisection Objective Function
    def BFObjective(alpha : float) -> float:
        x = x_start + alpha * x_diff
        
        # Build the linear system
        rhs = np.zeros(M+2); rhs[M+1] = 1.0
        def bordered_matvec(w : np.ndarray) -> np.ndarray: 
            z = w[0:M+1]; beta = w[M+1]
            Jz = (F(x + rdiff*z) - F(x - rdiff*z)) / (2*rdiff)
            J_eq = Jz + beta * r
            l_eq = np.dot(l, z)
            return np.append(J_eq, [l_eq]) - rhs

        # Solve the linear system to obtain beta = z_solution[-1]
        z_solution = opt.newton_krylov(bordered_matvec, z0, rdiff=rdiff)
        print('Linear residual', np.linalg.norm(bordered_matvec(z_solution)))
        beta = z_solution[M+1]

        return beta
    
    # Solve beta = 0. This is the location of the bifurcation point.
    try:
        print(BFObjective(-5.0), BFObjective(5.0))
        alpha_singular, result = opt.brentq(BFObjective, -5.0, 5.0, full_output=True, disp=False)
    except ValueError: # No sign change detected
        return False, x_end, 1.0
    except opt.NoConvergence:
        return False, x_end, 1.0
    x_singular = x_start + alpha_singular * x_diff

    return True, x_singular, alpha_singular