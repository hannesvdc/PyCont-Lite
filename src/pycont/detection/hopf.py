import numpy as np

from .base import DetectionModule, ObjectiveType
from ..Logger import LOG
from ..exceptions import InputError
from ._hopf import initializeHopf, refreshHopf, detectHopf

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

@dataclass
class HopfState:
    x : np.ndarray
    eigvals : np.ndarray
    eigvecs : np.ndarray
    lead : int

class HopfDetectionModule(DetectionModule):

    def __init__(self,
                 G: ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp: Dict[str, Any]) -> None:
        super().__init__("HB", G, u0, p0, sp)

        if self.M < 2:
            raise InputError(f"Can't do Hopf detection on one-dimensional systems.")
        self.n_hopf_eigenvalues = sp.get("n_hopf_eigenvalues", min(6, self.M))
        LOG.verbose(f'Hopf detector {self.n_hopf_eigenvalues}.')

    def initializeBranch(self,
                         x : np.ndarray,
                         tangent : np.ndarray) -> None:
        eigvals, eigvecs, lead = initializeHopf(self.G, x[0:self.M], x[self.M], self.n_hopf_eigenvalues, self.sp)
        self.prev_state = HopfState(np.copy(x), eigvals, eigvecs, lead)

    def update(self,
               F : Callable[[np.ndarray], np.ndarray],
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        u_new = x_new[0:self.M]
        p_new = x_new[self.M]
        eigvals, eigvecs, lead = refreshHopf(self.G, u_new, p_new, self.prev_state.eigvals, self.prev_state.eigvecs, self.sp)
        self.new_state = HopfState(np.copy(x_new), eigvals, eigvecs, lead)

        # If we passed a Hopf point, return True for localization.
        is_hopf = detectHopf(self.prev_state.eigvals, self.new_state.eigvals, self.prev_state.lead, self.new_state.lead)
        if is_hopf:
            LOG.info(f"Hopf Point Detected near {x_new}.")
            return True
        
        # Else, update the internal state
        self.prev_state = self.new_state
        return False
    
    def localize(self) -> Optional[np.ndarray]:
        return self.new_state.x