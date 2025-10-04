import numpy as np

from .base import DetectionModule, ObjectiveType
from ._fold import computeFoldPoint

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
               F : Callable[[np.ndarray], np.ndarray],
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

    def localize(self) -> Optional[np.ndarray]:
        is_fold, x_fold, _ = computeFoldPoint(self.G, self.prev_x, self.new_x, self.prev_tangent, self.sp)
        if is_fold:
            return x_fold
        return None