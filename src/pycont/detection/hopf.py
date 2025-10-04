import numpy as np

from .base import DetectionModule, ObjectiveType
from ..Logger import LOG
from ..exceptions import InputError
from ._hopf import initializeHopf, refreshHopf, detectHopf

from typing import Callable, Dict, Optional, Any

class HopfDetectionModule(DetectionModule):

    def __init__(self,
                 G: ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp: Dict[str, Any]) -> None:
        super().__init__(G, u0, p0, sp)

        if self.M < 2:
            raise InputError(f"Can't do Hopf detection on one-dimensional systems.")
        self.n_hopf_eigenvalues = sp.get("n_hopf_eigenvalues", min(6, self.M))
        LOG.verbose(f'Hopf detector {sp["n_hopf_eigenvalues"]}.')

    def initializeBranch(self,
                         x : np.ndarray,
                         tangent : np.ndarray) -> None:
        self.prev_x = np.copy(x)
        self.prev_hopf_state = initializeHopf(self.G, x[0:self.M], x[self.M], self.sp)

    def update(self,
               F : Callable[[np.ndarray], np.ndarray],
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        self.new_x = np.copy(x_new)
        self.new_hopf_state = refreshHopf(self.G, x_new[0:self.M], x_new[self.M], self.prev_hopf_state, self.sp)

        # If we passed a Hopf point, return True for localization.
        if detectHopf(self.prev_hopf_state, self.new_hopf_state):
            return True
        
        # Else, update the internal state
        self.prev_x = self.new_x
        self.prev_hopf_state = self.new_hopf_state
        return False
    
    def localize(self) -> Optional[np.ndarray]:
        pass