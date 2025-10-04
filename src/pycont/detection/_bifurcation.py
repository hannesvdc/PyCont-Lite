import numpy as np
import scipy.optimize as opt

from ..Logger import LOG

from typing import Callable, Tuple, Dict

def test_fn_jacobian(F : Callable[[np.ndarray], np.ndarray], 
					 x : np.ndarray,
					 l : np.ndarray, 
					 r : np.ndarray, 
					 w_prev : np.ndarray, 
					 sp : Dict) -> Tuple[np.ndarray, float]:
    """
    Bifurcation point test function. A bifurcation point is given by the scalar 
    `beta = 1/ l^T w' changing sign, where `w` is the solution to `J w = r` with
    `J` the Jacobian of the extended objective function `F`. 

    The Jacobian system is first solved using L-GMRES and refined with newton_krylov
    when the initial residual is too large.

    Parameters
    ----------
    F : Callable
        The extended objective function.
    x : ndarray
        The current point `(u,p)` on the branch.
    l, r : ndarray 
        The left and right test vectors. Cannot have any components in the direction of 
        the current tangent. Must also be normalized.
    w_prev : ndarray
        Solution to the Jacobian system in the previous point on the branch. Used as initial guess.
    sp : Dict
        Solver parameters.

    Returns
    -------
        w_solution : ndarray
            The full solution to the Jacobian system.
        beta : float
            The value of the test function. Monitor this for sign changes.
    """
    rdiff = sp["rdiff"]
    def matvec(w):
        norm_w = np.linalg.norm(w)
        if norm_w == 0.0:
            return -r
        eps = rdiff / norm_w
        return (F(x + eps * w) - F(x - eps * w)) / (2.0*eps) - r

    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        w_solution = opt.newton_krylov(matvec, w_prev, rdiff=rdiff, verbose=False)
    residual = np.linalg.norm(matvec(w_solution))
    beta = -1.0 / np.dot(l, w_solution)
    LOG.verbose(f'Jacobian test FN = {beta}, residual = {residual}')

    return w_solution, beta

def test_fn_jacobian_multi(F : Callable[[np.ndarray], np.ndarray], 
					       x : np.ndarray,
					       l_vectors : np.ndarray, 
					       r_vectors : np.ndarray, 
					       prev_w_vectors : np.ndarray, 
					       sp : Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    The main bifurcation detection test function.

    Parameters
    ----------
    F : Callable
        The extended objective function.
    x : ndarray
        The new point on the branch.
    l_vectors : ndarray
        The left vectors used for bifurcation detection.
    r_vectors : ndarray
        The right vectors used for bifurcation detection.
    prev_w_vectors : ndarray
        The bifurcation detection vectors at the previous points.
    sp : Dict
        Solver parameters

    Returns
    -------
    bifurcation_state : Dict
        Dictionary containing the current bifurcation detection state.
    """

    w_vectors = np.zeros_like(prev_w_vectors)
    w_values = np.zeros(prev_w_vectors.shape[0])
    for index in range(r_vectors.shape[0]):
        w_i, value_i = test_fn_jacobian(F, x, l_vectors[index], r_vectors[index], prev_w_vectors[index], sp)
        w_vectors[index,:] = w_i
        w_values[index] = value_i
    
    return w_values, w_vectors

def computeBifurcationPoint(F : Callable[[np.ndarray], np.ndarray],
                            x_left : np.ndarray,
                            x_right : np.ndarray,
							l_vectors : np.ndarray, 
							r_vectors : np.ndarray,
                            w_vectors_left : np.ndarray,
                            index : int,
                            M : int,
							sp : Dict) -> Tuple[bool, np.ndarray, float]:
    """
    Function that localizes the bifurcation point, if any, within a given tolerance.

    Parameters
    ----------
    F : Callable
        The extended objective function.
    x_left, x_right : ndarray
        Points to the left and right of the bifurcation point, respectively
    l_vectors, r_vectors : ndarray 
        The left and right test vectors.
    w_vectors_left : ndarray
        The w-vectors at x_left, used to speed up calculations.
    index : int
        The index of the test function that underwent a sign change.
    M : int
        The dimension of the state vector `u`.
    sp : Dict
        Solver parameters.

    Returns
    -------
        is_bf_point : bool
            True if we actually found a bifurcation point, False otherwise. Latter happens when the brentq
            optimizer did not converge.
        x_singular : ndarray
            The location of the bifurcation point, if converged.
        alpha_singular : float
            Fraction of the arc length between x_start and x_end where the bifurcation point lies.
    """
    rdiff = sp["rdiff"]

    x_diff = x_left - x_right
    l = l_vectors[index]
    r = r_vectors[index]
    w = w_vectors_left[index,:]
    if len(w) == M+1:
        S = np.dot(l, w)
        z0 = np.append(w / S, -1.0 / S)
    else:
        z0 = np.copy(w)

    # Build the Bisection Objective Function
    def BFObjective(alpha : float) -> float:
        x = x_left + alpha * x_diff
        
        # Build the linear system
        rhs = np.zeros(M+2); rhs[M+1] = 1.0
        def bordered_matvec(w : np.ndarray) -> np.ndarray: 
            z = w[0:M+1]; beta = w[M+1]
            Jz = (F(x + rdiff*z) - F(x - rdiff*z)) / (2*rdiff)
            J_eq = Jz + beta * r
            l_eq = np.dot(l, z)
            return np.append(J_eq, [l_eq]) - rhs

        # Solve the linear system to obtain beta = z_solution[-1]
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            z_solution = opt.newton_krylov(bordered_matvec, z0, rdiff=rdiff)
        LOG.verbose(f'Linear Bifurcation residual {np.linalg.norm(bordered_matvec(z_solution))}')
        beta = z_solution[M+1]

        return beta
    
    # Solve beta = 0. This is the location of the bifurcation point.
    try:
        LOG.verbose(f'BrentQ edge values {BFObjective(0.0)},  {BFObjective(1.0)}')
        alpha_singular, result = opt.brentq(BFObjective, 0.0, 1.0, full_output=True, disp=False)
    except ValueError: # No sign change detected
        LOG.verbose('Value error caught')
        return False, x_right, 1.0
    except opt.NoConvergence:
        LOG.verbose('NoConvergence error caught')
        return False, x_right, 1.0
    x_singular = x_left + alpha_singular * x_diff

    return True, x_singular, alpha_singular