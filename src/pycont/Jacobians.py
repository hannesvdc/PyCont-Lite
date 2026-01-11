import numpy as np
import scipy.sparse.linalg as slg

import abc
from typing import Callable, Any, Optional

class AbstractJacobian(abc.ABC):
    """ Abstract base class for all Jacobian types. This class provides four methods:
        - `getJacobianAt(u, p)` returns a Jacobian linear operator dG/du at (u, p).
        - `getParameterDerivativeAt(u, p)` returns the derivative of G to p at (u, p).
        - `getJacobianFunctional()` returns the Jacobian constructor function or None.
        - `getParameterDerivativeFunctional()` returns the paramter derivative constructor function.
    """

    @abc.abstractmethod
    def getJacobianAt(self,
                      u : np.ndarray,
                      p : float) -> slg.LinearOperator:
        pass

    @abc.abstractmethod
    def getParameterDerivativeAt(self, 
                                 u : np.ndarray,
                                 p : float) -> np.ndarray:
        pass

    def getJacobianFunctional(self) -> Optional[Callable[[np.ndarray, float], slg.LinearOperator]]:
        return None
    

class FullJacobian (AbstractJacobian):
    def __init__(self,
                 n : int,
                 Gu : Callable[[np.ndarray, float], Any],
                 Gp : Callable[[np.ndarray, float], np.ndarray]) -> None:
        def innerJacobianFunctional(u : np.ndarray, p : float):
            J = Gu(u, p)
            matvec = lambda v : J @ v
            return slg.LinearOperator(shape=(n,n), matvec=matvec) # type: ignore
        self.jacobianFunctional = innerJacobianFunctional
        self.parameterDerivative = lambda u, p : Gp(u, p)

    def getJacobianAt(self, u: np.ndarray, p: float) -> slg.LinearOperator:
        return self.jacobianFunctional(u, p)
    
    def getParameterDerivativeAt(self, u: np.ndarray, p: float) -> np.ndarray:
        return self.parameterDerivative(u, p)
    
    def getJacobianFunctional(self) -> Optional[Callable[[np.ndarray, float], slg.LinearOperator]]:
        return self.jacobianFunctional
    

class MatrixVectorJacobian (AbstractJacobian):
    def __init__(self,
                 n : int,
                 Gu_v : Callable[[np.ndarray, float, np.ndarray], np.ndarray],
                 Gp : Callable[[np.ndarray, float], np.ndarray]) -> None:
        self.jacobianFunctional = lambda u, p: slg.LinearOperator(shape=(n,n), matvec=lambda v: Gu_v(u, p, v)) # type: ignore
        self.parameterDerivative = lambda u, p : Gp(u, p)

    def getJacobianAt(self, u: np.ndarray, p: float) -> slg.LinearOperator:
        return self.jacobianFunctional(u, p)
    
    def getParameterDerivativeAt(self, u: np.ndarray, p: float) -> np.ndarray:
        return self.parameterDerivative(u, p)
    
    def getJacobianFunctional(self) -> Optional[Callable[[np.ndarray, float], slg.LinearOperator]]:
        return self.jacobianFunctional

class MatrixFreeJacobian (AbstractJacobian):
    def __init__(self,
                 n : int,
                 G : Callable[[np.ndarray, float], np.ndarray],
                 rdiff : float) -> None:
        self.n = n
        self.G = G
        self.rdiff = rdiff

        self.parameterDerivative = lambda u, p : (self.G(u, p + rdiff) - self.G(u, p - rdiff)) / (2.0*rdiff)

    def getJacobianAt(self, u: np.ndarray, p: float) -> slg.LinearOperator:
        def matvec(v : np.ndarray) -> np.ndarray:
            norm_v = np.linalg.norm(v)
            if norm_v == 0.:
                return 0.0 * v
            eps = self.rdiff / norm_v
            return (self.G(u + eps * v, p) - self.G(u - eps * v, p)) / (2.0*eps)
        return slg.LinearOperator(shape=(self.n, self.n), matvec=matvec) # type: ignore
    
    def getParameterDerivativeAt(self, u: np.ndarray, p: float) -> np.ndarray:
        return self.parameterDerivative(u, p)