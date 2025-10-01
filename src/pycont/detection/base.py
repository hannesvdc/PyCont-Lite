import numpy as np

import abc
from typing import Callable, Dict, Any

ObjectiveType = Callable[[np.ndarray, float], np.ndarray]

class DetectionModule(abc.ABC):

    def __init__(self,
                 G : ObjectiveType,
                 u0 : np.ndarray,
                 p0 : float,
                 sp : Dict[str, Any]):
        """
        Initialize new detection module for the problem.

        Parameters
        ----------
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
               x_new : np.ndarray,
               tangent_new : np.ndarray) -> bool:
        """
        Update the detection module with a new point on the branch. Returns true if
        a special point was passed, False otherwise.

        Parameters
        ----------
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
    def localize(self) -> np.ndarray:
        """
        Localize the special point to high precision.

        Returns
        -------
        x_loc : ndarray
            Point on the branch where the detection function is exactly 0, up to the tolerance.
        """
        raise NotImplementedError