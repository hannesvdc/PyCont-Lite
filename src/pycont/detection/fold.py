import numpy as np
import scipy.optimize as opt

from .base import DetectionModule, ObjectiveType
from ..Tangent import computeTangent
from ..Logger import LOG

from typing import Dict, Callable, Any, Optional

class FoldDetectionModule(DetectionModule):

    def __init__(self,
                 G: ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp: Dict[str, Any]) -> None:
        super().__init__(G, u0, p0, sp)

    def initializeBranch(self,
                         x : np.ndarray,
                         tangent : np.ndarray) -> None:
        self.prev_x = np.copy(x)
        self.prev_tangent = np.copy(tangent)

    def update(self,
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        self.new_x = np.copy(x_new)
        self.new_tangent = np.copy(tangent_new)

        if self.new_tangent[self.M] * self.prev_tangent[self.M] < 0.0:
            return True
        
        # Update the internal state
        self.prev_x = self.new_x
        self.prev_tangent = self.new_tangent
        return False

    def localize(self) -> np.ndarray:
        """
        Localizes the bifurcation point between x_start and x_end using the bisection method.

        Parameters
        ----------
            G: Callable
                Objective function with signature ``G(u,p) -> ndarray``
            x_left : ndarray 
                Starting point (u, p) to the 'left' of the fold point.
            x_right : ndarray 
                End point (u, p) to the 'right' of the fold point.
            tangent_ref : ndarray
                A reference tangent vector to speed up tangent calculations. Typically the 
                tangent vector at x_left.
            sp : Dict
                Solver parameters.

        Returns
        -------
            is_fold_point : boolean
                True if we detected an antual fold point.
            x_fold: ndarray
                The location of the fold point within the tolerance.
        """
        rdiff = self.sp["rdiff"]
        ds = np.linalg.norm(self.new_x - self.prev_x)
        tangent_ref = np.copy(self.prev_tangent)

        def make_F_ext(alpha : float) -> Callable[[np.ndarray], np.ndarray]:
            ds_alpha = alpha * ds
            N = lambda q: np.dot(tangent_ref, q - self.prev_x) - ds_alpha
            F = lambda q: np.append(self.G(q[0:self.M], q[self.M]), N(q))
            return F
        def finalTangentComponent(alpha : float) -> Optional[np.ndarray]:
            F = make_F_ext(alpha)
            try:
                with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
                    x_alpha = opt.newton_krylov(F, self.prev_x, rdiff=rdiff)
            except opt.NoConvergence:
                return None
            tangent = computeTangent(self.G, x_alpha[0:self.M], x_alpha[self.M], tangent_ref, self.sp)
            return tangent[self.M]

        try:
            LOG.info(f'BrentQ edge values {finalTangentComponent(-2.0)},  {finalTangentComponent(2.0)}')
            alpha_fold, result = opt.brentq(finalTangentComponent, -2.0, 2.0, full_output=True, disp=False)
        except ValueError: # No sign change detected
            return self.new_x

        x_fold = self.prev_x + alpha_fold * (self.new_x - self.prev_x)
        return x_fold