import numpy as np

from .base import DetectionModule, ObjectiveType
from ..Logger import LOG
from ..exceptions import InputError
from ._hopf import initializeHopf, refreshHopf, detectHopf, localizeHopf

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any

@dataclass
class HopfState:
    x : np.ndarray
    eigvals : np.ndarray
    eigvecs : np.ndarray
    lead : int

class HopfDetectionModule(DetectionModule):
    """
    Module to detect Hopf points by tracking the few eigenvalues with largest real part.
    The main assumption behind this implementation is that the system is only marginally
    unstable with only a few eigenvalues on the right side of the imaginary axis.

    Numerical eigenvalue computations and Hopf detection and localization are implemented in _hopf.py.
    """

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
        prev_lead_index = self.prev_state.lead
        prev_eigval = self.prev_state.eigvals[prev_lead_index]
        prev_eigvec = self.prev_state.eigvecs[:,prev_lead_index]
        lead_index = self.new_state.lead
        lead_eigval = self.new_state.eigvals[lead_index]
        lead_eigvec = self.new_state.eigvecs[:,lead_index]

        is_hopf, hopf_point = localizeHopf(self.G, self.prev_state.x, self.new_state.x, prev_eigval, lead_eigval, prev_eigvec, lead_eigvec, self.M, self.sp)
        if is_hopf:
            LOG.info(f'Hopf Point localized at {hopf_point}')
            return hopf_point
        
        LOG.info('Erroneous Hopf point detected, most likely due to inaccurate eigenvalue computations. Continuing along this branch.')
        self.prev_state = self.new_state
        return None