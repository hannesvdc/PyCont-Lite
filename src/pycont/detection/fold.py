import numpy as np

from .base import DetectionModule, ObjectiveType
from ._fold import computeFoldPoint
from ..Logger import LOG

from dataclasses import dataclass
from typing import Dict, Callable, Any, Optional

@dataclass
class FoldState:
    x : np.ndarray
    tangent : np.ndarray

class FoldDetectionModule(DetectionModule):

    def __init__(self,
                 G: ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp: Dict[str, Any]) -> None:
        super().__init__("LP", G, u0, p0, sp)

    def initializeBranch(self,
                         x : np.ndarray,
                         tangent : np.ndarray) -> None:
        self.prev_state = FoldState(np.copy(x), np.copy(tangent))

    def update(self,
               F : Callable[[np.ndarray], np.ndarray],
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        self.new_state = FoldState(np.copy(x_new), np.copy(tangent_new))

        if self.new_state.tangent[self.M] * self.prev_state.tangent[self.M] < 0.0:
            return True
        
        # Update the internal state
        self.prev_state = self.new_state
        return False

    def localize(self) -> Optional[np.ndarray]:
        is_fold, x_fold, _ = computeFoldPoint(self.G, self.prev_state.x, self.new_state.x, self.prev_state.tangent, self.sp)
        if is_fold:
            LOG.info(f'Fold point at {x_fold}')
            return x_fold
        
        LOG.info('Erroneous Fold Point detection due to blow-up in tangent vector.')
        self.prev_state = self.new_state
        return None