import numpy as np
import scipy.optimize as opt

from .base import DetectionModule, ObjectiveType
from ..Logger import LOG
from ..exceptions import InputError

from dataclasses import dataclass
from typing import Dict, Callable, Any, Tuple, Optional

@dataclass
class BifurcationState:
    x : np.ndarray
    tangent : np.ndarray
    w_values : np.ndarray
    w_vectors : np.ndarray

class BifurcationDetectionModule(DetectionModule):

    def __init__(self,
                 G: ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp: Dict[str, Any]) -> None:
        super().__init__(G, u0, p0, sp)

        # Do checks
        self.n_bifurcation_vectors = sp.setdefault("n_bifurcation_vectors", min(3, self.M))
        if self.n_bifurcation_vectors < 0:
            raise InputError(f"number of bifurcation vectors must be a positive integer, got {self.n_bifurcation_vectors}.")
        
    def _orthonormalize_lr(self,
                           tangent : np.ndarray) -> None:
        """
        Orthonormalize the rows of l_vectors and r_vectors (shape (k, M+1) each).
        Additionally, make each row of l_vectors orthogonal to the given tangent (length M+1).

        Parameters
        ----------
        tangent : ndarray
            The current tangent vector.
        """
        extended_r_vectors = self.r_vectors.T
        extended_l_vectors = np.concatenate((tangent[:,np.newaxis], self.l_vectors.T), axis=1)
        extended_r_vectors, _ = np.linalg.qr(extended_r_vectors, mode='reduced')
        extended_l_vectors, _ = np.linalg.qr(extended_l_vectors, mode='reduced')

        self.l_vectors =  extended_l_vectors[:,1:].T
        self.r_vectors = extended_r_vectors.T

    def initializeBranch(self,
                         x : np.ndarray,
                         tangent : np.ndarray) -> None:
        
        # Sample the l- and r-vectors for this test function.
        rng = np.random.RandomState(seed=self.sp["seed"])
        self.r_vectors = rng.normal(0.0, 1.0, (self.n_bifurcation_vectors, self.M+1))
        self.l_vectors = rng.normal(0.0, 1.0, (self.n_bifurcation_vectors, self.M+1))
        self._orthonormalize_lr(tangent)

        # Initialize the BifurcationState along this branch
        init_w_vectors = np.zeros_like(self.r_vectors)
        init_w_values = np.zeros(self.n_bifurcation_vectors)
        self.prev_state = BifurcationState(np.copy(x), np.copy(tangent), init_w_values, init_w_vectors)

    def update(self,
               F : Callable[[np.ndarray], np.ndarray],
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:

        # Calculate the test function for every l- and r-vector
        w_vectors = np.zeros_like(self.prev_state.w_vectors)
        w_values = np.zeros_like(self.prev_state.w_values)
        for index in range(len(w_values)):
            w_i, value_i = self.test_fn_jacobian(F, x_new, index)
            w_vectors[index,:] = w_i
            w_values[index] = value_i
        self.new_state = BifurcationState(np.copy(x_new), np.copy(tangent_new), w_values, w_vectors)

        # Test for a bifurcation point
        is_bf =  (w_values * self.prev_state.w_values < 0.0) & (np.abs(self.prev_state.w_values) < 1000.0) & (np.abs(w_values) < 1000.0)
        if is_bf:
            self.F_bf = F
            return True
        
        # Else update the internal state
        self.prev_state = self.new_state
        return False
    
    def test_fn_jacobian(self, 
                         F : Callable[[np.ndarray], np.ndarray], 
					     x : np.ndarray,
                         index : int) -> Tuple[np.ndarray, float]:
        # Gather required state
        rdiff = self.sp["rdiff"]
        r = self.r_vectors[index,:]
        w_prev = self.prev_state.w_vectors[index,:]

        def matvec(w):
            norm_w = np.linalg.norm(w)
            if norm_w == 0.0:
                return -r
            eps = rdiff / norm_w
            return (F(x + eps * w) - F(x - eps * w)) / (2.0*eps) - r

        # Compute test-function value
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            w_solution = opt.newton_krylov(matvec, w_prev, rdiff=rdiff, verbose=False)
        residual = np.linalg.norm(matvec(w_solution))
        beta = -1.0 / np.dot(self.l_vectors[index,:], w_solution)
        LOG.verbose(f'Jacobian test FN = {beta}, residual = {residual}')

        # return the full vector and the test function value
        return w_solution, beta
    
    def localize(self) -> Optional[np.ndarray]:
        rdiff = self.sp["rdiff"]

        # Gather the required state
        x_start = self.prev_state.x
        x_end = self.new_state.x
        index = np.argwhere(self.prev_state.w_values * self.new_state.w_values < 0.0)[0]
        w_vector = self.new_state.w_vectors[index, :]
        r = self.r_vectors[index,:]
        l = self.l_vectors[index,:]

        # Initial condition for Newton
        x_diff = x_end - x_start
        S = np.dot(l, w_vector)
        z0 = np.append(w_vector / S, -1.0 / S)

        # Build the Bisection Objective Function
        def BFObjective(alpha : float) -> float:
            x = x_start + alpha * x_diff
            
            # Build the linear system
            rhs = np.zeros(self.M+2); rhs[self.M+1] = 1.0
            def bordered_matvec(w : np.ndarray) -> np.ndarray: 
                z = w[0:self.M+1]; beta = w[self.M+1]
                Jz = (self.F_bf(x + rdiff*z) - self.F_bf(x - rdiff*z)) / (2*rdiff)
                J_eq = Jz + beta * r
                l_eq = np.dot(l, z)
                return np.append(J_eq, [l_eq]) - rhs

            # Solve the linear system to obtain beta = z_solution[-1]
            with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                z_solution = opt.newton_krylov(bordered_matvec, z0, rdiff=rdiff)
            LOG.verbose(f'Linear Bifurcation residual {np.linalg.norm(bordered_matvec(z_solution))}')
            beta = z_solution[self.M+1]

            return beta
        
        # Solve beta = 0. This is the location of the bifurcation point.
        try:
            LOG.verbose(f'BrentQ edge values {BFObjective(0.0)},  {BFObjective(1.0)}')
            alpha_singular, result = opt.brentq(BFObjective, 0.0, 1.0, full_output=True, disp=False)
        except ValueError: # No sign change detected
            LOG.verbose('Value error caught')
            return None
        except opt.NoConvergence:
            LOG.verbose('NoConvergence error caught')
            return None
        x_singular = x_start + alpha_singular * x_diff

        return x_singular