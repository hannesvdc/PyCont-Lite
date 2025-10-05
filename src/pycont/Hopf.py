import numpy as np
import scipy.sparse.linalg as slg

from .Logger import LOG

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
                   sp: Dict) -> Dict:
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
    sp : Dict
        Solver parameters including arguments `keep_r` and `m_target`.

    Returns
    -------
    hopf_state: Dict
        Contains the current eigenvalues "eig_vals" and eigenvectors "eig_vecs", index 
        "lead" of the eigenvalue  closes to the imaginary axis, and "omega" the imaginary
        part of this eigenvalue.
    """
    LOG.verbose(f"Initializing Hopf")
    m_eigs = sp["n_hopf_eigenvalues"]
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

    omega = float(abs(eigvals[lead].imag)) if lead != -1 else 0.0
    state = {
        "eig_vals" : eigvals, "eig_vecs" : V, "lead" : lead, "omega" : omega
    }
    return state

def refreshHopf(G: Callable[[np.ndarray, float], np.ndarray],
                u : np.ndarray,
                p : float,
                prev_hopf_state : Dict,
                sp: Dict) -> Dict:
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
    prev_hopf_state : Dict
        The Hopf state at the previous continuation point.
    sp : Dict
        Solver parameters including arguments `keep_r` and `m_target`.

    Returns
    -------
    hopf_state: Dict
        Contains the current eigenvalues "eig_vals" and eigenvectors "eig_vecs", index 
        "lead" of the eigenvalue  closes to the imaginary axis, and "omega" the imaginary
        part of this eigenvalue.
    """
    jitter = 0.001
    omega_min = 1e-3

    eig_vals_prev = prev_hopf_state["eig_vals"]
    eig_vecs_prev = prev_hopf_state["eig_vecs"]
    eig_vals_new = np.empty_like(eig_vals_prev, dtype=np.complex128)
    eig_vecs_new = np.empty_like(eig_vecs_prev, dtype=np.complex128)

    # Create JVP
    M = len(u)
    rdiff = sp["rdiff"]
    Jv = lambda v: (G(u + rdiff*v, p) - G(u - rdiff*v, p)) / (2.0*rdiff)

    # Loop over previous eigenvalues and update with the new Jacobian
    for i, (sigma_i, v_i) in enumerate(zip(eig_vals_prev, eig_vecs_prev.T)):
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
        eig_vals_new[i] = sigma_new
        eig_vecs_new[:, i] = v_new

    # Pick lead complex eigenvalue closest to imaginary axis
    lead = _pick_near_axis(eig_vals_new, omega_min)  # returns -1 if none
    omega = float(abs(eig_vals_new[lead].imag)) if lead != -1 else 0.0
    LOG.verbose(f'Hopf Value {eig_vals_new[lead]}')

    return {"eig_vals": eig_vals_new, "eig_vecs": eig_vecs_new, "lead": lead, "omega": omega}

def detectHopf(prev_state : Dict,
               curr_state : Dict) -> bool:
    """
    Main Hopf detection algorith. Checks if the real parts of the leading eigenvalues
    in the state dicts have a different sign.

    Parameters
    ----------
    prev_state : Dict
        State of the Hopf detection function at the previous point.
    curr_state : Dict
        State of the Hopf detection function at the current point.

    Returns
    -------
    is_hopf : bool
        True if a Hopf point lies between the two points, False otherwise.
    """
    if prev_state["lead"] < 0 or curr_state["lead"] < 0:
        return False
    
    prev_leading_ritz_value = prev_state["eig_vals"][prev_state["lead"]]
    curr_leading_ritz_value = curr_state["eig_vals"][curr_state["lead"]]

    return np.real(prev_leading_ritz_value) * np.real(curr_leading_ritz_value) < 0.0