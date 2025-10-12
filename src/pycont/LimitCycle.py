import numpy as np
import scipy.optimize as opt

from .Logger import LOG

from typing import Callable, Dict, Tuple, Optional

def buildODEObjective(G : Callable[[np.ndarray, float], np.ndarray],
                      dtau : float,
                      M : int,
                      L : int) -> Callable[[np.ndarray,float,float],np.ndarray]:
    """ 
    Internal helper function that returns the limit cycle ODE objective function.
    It only requires the origianl objective function and time-discretization parameters.

    Parameters
    ----------
    G : Callable
        The original objective function
    dtau : float
        The normalized limit cycle time step.
    M : int
        Size of the state vector `u`.
    L : int
        Number of collocation points along the limit cycle.
    """
    def ODEObjective(U : np.ndarray, # one long array 
                     T : float, 
                     p : float) -> np.ndarray:
        # Reshape X into rows
        assert len(U) == M * L
        U = np.reshape(U, (L, M))
        U_shifted = np.roll(U, shift=1, axis=0)

        X_alpha = 0.5 * (U + U_shifted)
        G_alpha = G(X_alpha, p)  # Assumes G can be evaluated for every row.
        
        R = U_shifted - U - dtau * T * G_alpha
        return R.flatten()
    return ODEObjective

def createLimitCycleObjectiveFunction(G : Callable[[np.ndarray, float], np.ndarray],
                                      U_ref : np.ndarray,
                                      M : int,
                                      L : int = 16) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Internal function to create the objective function for limit cycle continuation, 
    starting from the initial (typically tiny) limit cycle `X_ref`.

    Parameters
    ----------
    G : Callable
        The original (steady-state) objective function. When the user wants limit
        cycle continuation, `G` should be able to take a matrix as its first argument
        and output the objective function for every row in the matrix.
    U_ref : np.ndarray
        The initial limit cycle on the branch, typically comes from `calculateInitialLimitCycle` below.
    M : int
        The size of the state variable `u`.
    L : int
        The number of collocation points on the limit cycle. As a default, we represent 
        the limit cycle in `L=16` points.

    Returns
    -------
    GLC : Callable
        Limit cycle objective function that inputs a M*L+1-dimensional array and 
        the paramter `p` outputs vector of the same size.

    """
    dtau = 1.0 / L
    U_ref = np.reshape(U_ref, (L, M))
    dU_ref_dtau = (np.roll(U_ref, shift=1, axis=0) - U_ref) / dtau
    
    # Build the Continuation objective function
    ODEObjective = buildODEObjective(G, dtau, M, L)
    def phaseCondition(U : np.ndarray) -> float:
        U = np.reshape(U, (L, M))
        dU_dtau = (np.roll(U, shift=1, axis=0) - U) / dtau

        inner_products  = np.sum(dU_dtau * dU_ref_dtau, axis=1)
        return np.sum(inner_products * dtau)
    def GLC(Q : np.ndarray,
            p : float):
        U = Q[:-1]
        T = Q[-1]
        return np.append(ODEObjective(U, T, p), phaseCondition(U))
    
    return GLC

def calculateInitialLimitCycle(G : Callable[[np.ndarray, float], np.ndarray],
                               sp : Dict,
                               x_hopf : np.ndarray,
                               omega : float,
                               eigvec : np.ndarray,
                               M : int,
                               L : int = 16,
                               rho : float = 0.01) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    Calculate the initial limit cycle close to the Hopf bifurcation point.

    Parameters
    ----------
    G : Callable
        The original (steady-state) objective function. When the user wants limit
        cycle continuation, `G` should be able to take a matrix as its first argument
        and output the objective function for every row in the matrix.
    sp : Dict
        Solver parameters.
    x_hopf : ndarray
        The Hopf point `(u_hopf, p_hopf)`.
    omega : float
        The imaginary part of the Hopf eigenvalue - real part should be 0.
    eigvec : ndarray[complex128]. 
        The associated eigenvector to eigenvalue `0 + i * omega`.
    M : int
        The dimension of the state vector `u`.
    L : int
        The number of collocation points along the limit cyle. Default 16.
    rho : float
        Initial offset from the Hopf point used to seed the Newton-Krylov method.
        Default is 1e-2.

    Returns
    -------
    U_LC : ndarray - matrix.
        The state vectors along the limit cycle appended into one long vector of size `L*M`.
    T_LC : float
        The initial period of the limit cycle.
    p_LC : float
        The initial parameter value along the limit cycle branch.
    """
    u_hopf = x_hopf[0:M]
    p_hopf = x_hopf[M]

    # Orthogonalize the real and imaginary components of `eigvec`.
    qr = np.real(eigvec)
    qi = np.imag(eigvec)
    qr = qr / np.linalg.norm(qr)
    qi = qi - np.dot(qr, qi) * qr
    qi /= np.linalg.norm(qi)
    
    # Build the objective function and initial phase condition
    dtau = 1.0 / L
    ODEObjective = buildODEObjective(G, dtau, M, L)
    def phaseCondition(U : np.ndarray) -> float:
        U0 = U[0:M]
        return np.dot(qr, U0 - u_hopf) - rho
    def initialObjective(Q : np.ndarray, p : float) -> np.ndarray:
        U = Q[0:-1]
        T = Q[-1]
        return np.append(ODEObjective(U, T, p), phaseCondition(U))
    
    # Compute the initial guess for the limit cycle
    tau = np.arange(L) / L
    T_init = 2.0 * np.pi / omega
    U_init = u_hopf[np.newaxis,:] + rho * (np.outer(np.cos(2.0*np.pi*tau), qr) - np.outer(np.sin(2.0*np.pi*tau), qi))
    Q_init = np.append(U_init, T_init)

    # Try positive guess first
    p_init = p_hopf + rho**2
    try:
        QLC = opt.newton_krylov(lambda Q : initialObjective(Q, p_init), Q_init, rdiff=sp["rdiff"], f_tol=sp["tolerance"])
        return QLC[:-1], QLC[-1], p_init
    except opt.NoConvergence:
        LOG.verbose('Initial guess for the limit cycle failed. Trying different sign of p.')
        
    # If it failed, try a negative guess
    p_init = p_hopf - rho**2
    try:
        QLC = opt.newton_krylov(lambda Q : initialObjective(Q, p_init), Q_init, rdiff=sp["rdiff"], f_tol=sp["tolerance"])
    except opt.NoConvergence:
        LOG.info('Initializing the Limit Cylce failed. Not doing running continuation on this branch.')
        return None

    # Return the initial limit cycle parameters
    return QLC[:-1], QLC[-1], p_init