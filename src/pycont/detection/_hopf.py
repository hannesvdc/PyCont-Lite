import numpy as np
import scipy.sparse.linalg as slg

from ..Logger import LOG

from typing import Callable, Dict, Tuple

def _pick_near_axis(vals: np.ndarray, omega_min: float) -> int:
    """
    Return the index of the complex eigenvalue (|Im| > omega_min)
    closest to the imaginary axis (min |Re|).

    Returns -1 if none qualify.
    """
    vals = np.asarray(vals, dtype=np.complex128)
    mask = np.where(np.abs(np.imag(vals)) > omega_min)[0]
    if mask.size == 0:
        return -1
    return int(mask[np.abs(np.real(vals[mask])).argmin()])

def _filterComplexConjugated(eigvals: np.ndarray,
                             eigvecs: np.ndarray,
                             omega_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep:
      - all ~real eigenvalues (|Im(λ)| <= omega_min)
      - complex eigenvalues with strictly positive imaginary part (Im(λ) > omega_min)
    Drop the corresponding negative-imaginary partners.
    Returns filtered eigenvalues and matching eigenvectors (columns).

    Parameters
    ----------
    eigvals : (m,) complex ndarray
    eigvecs : (n, m) complex ndarray
        Eigenvectors in columns aligned with eigvals.
    omega_min : float
        Imag threshold to consider Im(λ) ~ 0 (i.e., real).

    Returns
    -------
    vals_out : (k,) complex ndarray
    vecs_out : (n, k) complex ndarray
    """
    # Real if |Im| ≤ omega_min; complex+ if Im > omega_min
    real_mask = np.abs(np.imag(eigvals)) <= omega_min
    pos_imag_mask = np.imag(eigvals) > omega_min
    keep_mask = real_mask | pos_imag_mask

    vals_out = eigvals[keep_mask]
    vecs_out = eigvecs[:, keep_mask]
    return vals_out, vecs_out

def initializeHopf(G: Callable[[np.ndarray, float], np.ndarray],
                   u : np.ndarray,
                   p : float,
                   m_eigs : int,
                   sp: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Initialize the Hopf Bifurcation Detection Method by generating the eigenvalues 
    closest to the imaginary axis. These are the ones we want to follow throughout
    the arclength continuation method. This method assumes that only a few
    eigenvalues are unstable, i.e., right of the imaginary axis. Then we can rely
    on scipy.sparse.linalg.eigs to compute the eigenvalues using `which='LR'`. 

    Parameters
    ----------
    G : Callable
        The objective function.
    u : ndarray
        The current state vector on the path.
    p : float
        The current parameter value.
    m_eigs : int
        The number of Hopf eigenvalues to track.
    sp : Dict
        Solver parameters including arguments `keep_r` and `m_target`.

    Returns
    -------
    eigvals : ndarray
        Eigenvalues of DG with largest real part
    eigvecs : ndarray
        Corresponding eigenvectors in the columns of this matrix
    lead : int
        The index of the leading eigenvalue - the one closest to the imaginary axis.
    """
    LOG.verbose(f"Initializing Hopf")
    omega_min = 1e-3

    # Create JVP
    M = len(u)
    rdiff = sp["rdiff"]
    Jv = lambda v: (G(u + rdiff*v, p) - G(u - rdiff*v, p)) / (2.0*rdiff)

    # Compute the initial seed of many eigenvectors with largest real part (see assumption).
    if M > 2:
        k_pool = min(m_eigs, max(1, M-2))
        A = slg.LinearOperator(shape=(M, M), 
                               matvec=lambda v: Jv(v.astype(np.complex128, copy=False)), # type:ignore
                               dtype=np.complex128)
        eigvals, V = slg.eigs(A, k=k_pool, which="LR", return_eigenvectors=True) # type: ignore[reportAssignmentType]
    elif M == 2: # edge case M = 2. Compute eigenvalues explicitly
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        J  = np.column_stack((Jv(e1), Jv(e2)))
        eigvals, V = np.linalg.eig(J)

    # Pick the lead eigenvalue and return a Hopf state
    eigvals, V = _filterComplexConjugated(eigvals, V, omega_min)
    lead = _pick_near_axis(eigvals, omega_min)
    if lead != -1 and np.abs(np.real(eigvals[lead])) < 1e-10:
        eigvals[lead] = 1j * np.imag(eigvals[lead])
    LOG.verbose(f'eigvals{eigvals}')

    return eigvals, V, lead

def refreshHopf(G: Callable[[np.ndarray, float], np.ndarray],
                u : np.ndarray,
                p : float,
                eigvals_prev : np.ndarray,
                eigvecs_prev : np.ndarray,
                sp: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Recompute Hopf state by updating the eigenvalues closest to the imaginary axis. 
    Updating is done by one iteration of the Rayleigh method: for each eigenpair
    (sigma_i, vi) at the previous point, we solve (J - simga_i I) v = v_i and compute
    the new eigenvalue as the Rayleigh coefficient <J v, v>. 

    Parameters
    ----------
    G : Callable
        The objective function.
    u : ndarray
        The current state vector on the path.
    p : float
        The current parameter value.
    eigvals_prev : ndarray
        The eigenvalues at the previous continuation piont.
    eigvecs_prev : ndarray
        The eigenvectors at the previous continuation point.
    sp : Dict
        Solver parameters including arguments `keep_r` and `m_target`.

    Returns
    -------
    eigvals_new : ndarray
        Eigenvalues of DG with largest real part at `(u,p)`
    eigvecs_new : ndarray
        Corresponding eigenvectors in the columns of this matrix
    lead : int
        The index of the leading eigenvalue - the one closest to the imaginary axis.
    """
    jitter = 0.001
    omega_min = 1e-3

    eigvals_new = np.empty_like(eigvals_prev, dtype=np.complex128)
    eigvecs_new = np.empty_like(eigvecs_prev, dtype=np.complex128)

    # Create JVP
    M = len(u)
    rdiff = sp["rdiff"]
    Jv = lambda v: (G(u + rdiff*v, p) - G(u - rdiff*v, p)) / (2.0*rdiff)

    # Loop over previous eigenvalues and update with the new Jacobian
    for i, (sigma_i, v_i) in enumerate(zip(eigvals_prev, eigvecs_prev.T)):
        v0 = v_i.astype(np.complex128, copy=False)
        nv = np.linalg.norm(v0)
        v0 = v0 / nv

        # define (J - sigma I) operator, with a tiny imaginary jitter for stability
        shift = sigma_i + 1j * jitter
        def A_mv(x):
           x = x.astype(np.complex128, copy=False)
           return Jv(x) - shift * x
        A = slg.LinearOperator(shape=(M, M), matvec=A_mv, dtype=np.complex128) # type:ignore

        #inexact solve: (J - sigma I) w = v0
        w, info = slg.lgmres(A, v0, maxiter=8)
        residual = np.linalg.norm(A_mv(w) - v0)
        v_new = w / (np.linalg.norm(w) + 1e-16)
        LOG.verbose(f'Hopf LGRMES Resisdual {residual}')

        # Rayleigh quotient update
        Jv_v_new = Jv(v_new)
        sigma_new = np.vdot(v_new, Jv_v_new) / np.vdot(v_new, v_new)
        eigvals_new[i] = sigma_new
        eigvecs_new[:, i] = v_new

    # Pick lead complex eigenvalue closest to imaginary axis
    lead = _pick_near_axis(eigvals_new, omega_min)  # returns -1 if none
    LOG.verbose(f'Hopf Value {eigvals_new[lead]}')

    return eigvals_new, eigvecs_new, lead

def detectHopf(eigvals_prev : np.ndarray,
               eigvals_new : np.ndarray,
               lead_prev : int,
               lead_new) -> bool:
    """
    Main Hopf detection algorith. Checks if the real parts of the leading eigenvalues
    in the state dicts have a different sign.

    Parameters
    ----------
    eigvals_prev : Dict
        Hopf eigenvalues at the previous point.
    eigvals_new : Dict
        Hopf eigenvalues at the current point.
    lead_prev : int
        Index of the leading eigenvalue in `eigvals_prev`.
    lead_new : int 
        Index of the leading eigenvalue in `eigvals_new`.

    Returns
    -------
    is_hopf : bool
        True if a Hopf point lies between the two points, False otherwise.
    """
    if lead_prev < 0 or lead_new < 0:
        return False
    prev_leading_ritz_value = eigvals_prev[lead_prev]
    curr_leading_ritz_value = eigvals_new[lead_new]

    return np.real(prev_leading_ritz_value) * np.real(curr_leading_ritz_value) < 0.0