import numpy as np
import scipy.optimize as opt

from .base import DetectionModule, ObjectiveType

from typing import Dict, Any

class ParamMaxDetectionModule(DetectionModule):

    def __init__(self,
                 param_max_value : float) -> None:
        super().__init__()
        self.param_max_value = param_max_value

    def initializeBranch(self,
                         G: ObjectiveType,
                         x: np.ndarray,
                         tangent: np.ndarray,
                         sp: Dict[str, Any]) -> None:
        super().initializeBranch(G, x, tangent, sp)

        self.M = len(x) - 1
        self.u_prev = x[:self.M]
        self.p_prev = x[self.M]
    
    def update(self,
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        # Update the internal state
        self.u_new = x_new[:self.M]
        self.p_new = x_new[self.M]

        # Return true if we passed `param_max`. Otherwise update the internal state.
        if self.p_new >= self.param_max_value and self.p_prev < self.param_max_value:
            return True
        
        self.u_prev = self.u_new
        self.p_prev = self.p_new
        return False
    
    def localize(self) -> np.ndarray:
        alpha = (self.param_max_value - self.p_prev) / (self.p_new - self.p_prev)
        u_guess = self.u_prev + alpha * (self.u_new - self.u_prev)

        # Use Newton-Krylov to determine the exact point on the branch where `p = param_max`.
        objective = lambda u : self.G(u, self.param_max_value)
        rdiff = self.sp["rdiff"]
        tolerance = self.sp["tolerance"]
        try:
            u_param_max = opt.newton_krylov(objective, u_guess, rdiff=rdiff, f_tol=tolerance)
        except opt.NoConvergence as e:
            u_param_max = e.args[0]

        # Return the full state at param_max
        return np.append(u_param_max, self.param_max_value)
