import numpy as np

from ..Types import Event, EventKind

import abc
from typing import Callable, Dict, Any, Optional

ObjectiveType = Callable[[np.ndarray, float], np.ndarray]

class DetectionModule(abc.ABC):
    """
    Abstract base class for all detection modules. Each detection module should implement
    three methods
    - `initializeBranch(x, tangent)` : Calculate the initial state on a new branch given the initial point and its tangent.
    - `update(F, x_new, tangent_new)` : Update the internal state at `x_new` and return whether a special point was passed.
                                        Here, `F` is the extended objective
    - `localize()` : Compute the special (i.e. fold, bifurcation, ...) point up to the tolerance. 
                     Call signature might be subject to change.
    """

    def __init__(self,
                 kind : EventKind,
                 G : ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp : Dict[str, Any]):
        """
        Initialize new detection module for the problem.

        Parameters
        ----------
        kind : EventKind
            The corresponding event kind to this detection module.
        G : Callable
            The continuation objective function.
        u0 : ndarray
            Numerical continuation starting point
        p0 : float
            Numerical continuation starting parameter value
        sp : Optional Dict
            The solver parameters.

        Returns
        -------
        Nothing.
        """
        self.kind : EventKind = kind

        self.G = G
        self.M = len(u0)
        self.sp = {} if sp is None else dict(sp)

    @abc.abstractmethod
    def initializeBranch(self,
                         x : np.ndarray,
                         tangent : np.ndarray) -> None:
        """
        Initialize detection on a new branch. This function should reset all fields
        within the DetectionModule subclass.

        Parameters
        ----------
        x : ndarray
            The initial point on the branch
        tangent : ndarray
            The tangent to the branch at the inital point.

        Returns
        -------
        Nothing.
        """

    @abc.abstractmethod
    def update(self,
               F : Callable[[np.ndarray], np.ndarray],
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        """
        Update the detection module with a new point on the branch. Returns true if
        a special point was passed, False otherwise.

        Parameters
        ----------
        F : Callable
            Extended objective function.
        x_new : ndarray
            The new point on the branch.
        tangent_new : ndarray
            Tangent at the new point.

        Returns
        -------
        passed_point : bool
            True if a special point was passed, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def localize(self) -> Optional[np.ndarray]:
        """
        Localize the special point to high precision.

        Returns
        -------
        x_loc : ndarray
            Point on the branch where the detection function is exactly 0, up to the tolerance.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def addTerminationInfo(self, event : Event) -> Event:
        """
        Add important information to the branch termination event that 
        is used in further calculations.

        Returns
        -------
        Event
            The termination event with extra information.
        """
        raise NotImplementedError