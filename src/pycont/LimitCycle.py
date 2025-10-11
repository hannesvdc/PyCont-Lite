import numpy as np

from typing import Callable, Dict

def createLimitCycleObjectiveFunction(G : Callable[[np.ndarray, float], np.ndarray],
                                      X_ref : np.ndarray,
                                      M : int,
                                      L : int) -> Callable[[np.ndarray, float], np.ndarray]:
    dtau = 1.0 / L
    X_ref = np.reshape(X_ref, (L, M))
    dX_ref_dtau = (np.roll(X_ref, shift=1, axis=0) - X_ref) / dtau
    
    # Build the ODE Solution system
    def ODEObjective(X : np.ndarray, # one long array 
                     T : float, 
                     p : float) -> np.ndarray:
        # Reshape X into rows
        assert len(X) == M * L
        X = np.reshape(X, (L, M))
        X_shifted = np.roll(X, shift=1, axis=0)

        X_alpha = 0.5 * (X + X_shifted)
        G_alpha = G(X_alpha, p)  # Assumes G can be evaluated for every row.
        
        R = X_shifted - X - dtau * T * G_alpha
        return R.flatten()
    
    # Build the phase condition
    def phaseCondition(X : np.ndarray) -> float:
        X = np.reshape(X, (L, M))
        dX_dtau = (np.roll(X, shift=1, axis=0) - X) / dtau

        inner_products  = np.sum(dX_dtau * dX_ref_dtau, axis=1)
        return np.sum(inner_products * dtau)
    
    # Combine into one objective.
    def GLC(Q : np.ndarray,
            p : float):
        X = Q[:-1]
        T = Q[-1]
        return np.append(ODEObjective(X, T, p), phaseCondition(X))
    return GLC