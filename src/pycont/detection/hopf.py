import numpy as np

from .base import DetectionModule, ObjectiveType
from ..Logger import LOG
from ..exceptions import InputError
from ._hopf import initializeHopf, refreshHopf, detectHopf

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

@dataclass
class HopfState:
    eigvals : np.ndarray
    eigvecs : np.ndarray
    lead : int

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
        eigvals, eigvecs, lead = initializeHopf(self.G, x[0:self.M], x[self.M], self.sp)
        self.prev_state = HopfState(eigvals, eigvecs, lead)

    def update(self,
               F : Callable[[np.ndarray], np.ndarray],
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        self.new_x = np.copy(x_new)
        eigvals, eigvecs, lead = refreshHopf(self.G, x_new[0:self.M], x_new[self.M], self.prev_state.eigvals, self.prev_state.eigvecs, self.sp)
        self.new_state = HopfState(eigvals, eigvecs, lead)

        # If we passed a Hopf point, return True for localization.
        if detectHopf(self.prev_state.eigvals, self.new_state.eigvecs, self.prev_state.lead, self.new_state.lead):
            return True
        
        # Else, update the internal state
        self.prev_x = self.new_x
        self.prev_state = self.new_state
        return False
    
    def localize(self) -> Optional[np.ndarray]:
        pass