import numpy as np
import scipy.optimize as opt

from .base import DetectionModule, ObjectiveType

from typing import Dict, Any

class ParamMinDetectionModule(DetectionModule):

    def __init__(self,
                 param_min_value : float) -> None:
        super().__init__()
        self.param_min_value = param_min_value

    def initializeBranch(self,
                         G: ObjectiveType,
                         x: np.ndarray,
                         tangent: np.ndarray,
                         sp: Dict[str, Any]) -> None:
        super().initializeBranch(G, x, tangent, sp)

        self.M = len(x) - 1
        self.x = np.copy(x)
        self.p = self.x[self.M]
    
    def update(self,
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        self.x = np.copy(x_new)
        self.p = x_new[self.M]
        return (self.p <= self.param_min_value)

    def localize(self) -> np.ndarray:
        # Use Newton-Krylov to determine the exact point
        # on the branch where p = param_min.
        u0 = self.x[:self.M]
        objective = lambda u : self.G(u, self.param_min_value)
        rdiff = self.sp["rdiff"]
        tolerance = self.sp["tolerance"]
        u_param_min = opt.newton_krylov(objective, u0, rdiff=rdiff, f_tol=tolerance)

        return np.append(u_param_min, self.param_min_value)
