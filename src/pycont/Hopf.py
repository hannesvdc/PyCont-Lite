import numpy as np
import scipy.sparse.linalg as slg

from .Logger import LOG

from typing import Callable, Dict, Tuple

def _orthonormalize(X: np.ndarray) -> np.ndarray:
    """
    Internal function that orthonormalizes the columns of matrix X.

    Parameters
    ----------
    X : ndarray (M+1, n_input_vecs)
        Matrix with vectors in the columns.

    Returns
    -------
    V : ndarray (M+1, n_output_vecs)
        Matrix with orthonormal columns. `n_output_vecs <= n_input_vecs` is guarenteed. 
        Inequality is strict when two or more columns of `X` are linearly dependent.
    """
    column_index = 0
    V = np.zeros_like(X)
    for j in range(X.shape[1]):
        w = X[:, j]
        for i in range(column_index):
        #for v in V:
            w -= np.vdot(V[:,i], w) * V[:,i]
        norm_w = np.sqrt(np.vdot(w, w))
        if norm_w > 0:
            V[:,column_index] = w / norm_w
            column_index += 1
    return V[:,:column_index]

def arnoldi_from_seed(Jv: Callable[[np.ndarray], np.ndarray],
                      V0: np.ndarray, 
                      m_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct `m_target` orthonormal vectors using the Arnoldi method
    starting from the columns of `V0` as seed. 

    Parameters
    ----------
    Jv : Callable
        Jacobian-vector product of the objective function `F`.
    V0 : ndarray
        Initial orthonormal vectors as columns used as seed.
    m_target : int
        Total number of orthonormal vectors to build.

    Returns
    -------
    V : ndarray
        Matrix with orthonormal columns created by applying `Jv` to `V0` repeatedly.
    H : ndarrray
        Square upper Hessenberg matrix, obtained for free from the Arnoldi method.
    """
    dtype = V0.dtype
    r_keep = V0.shape[1]
    V = np.zeros((V0.shape[0], m_target), dtype=dtype)
    V[:,:r_keep] = V0
    H = np.zeros((m_target + 1, m_target), dtype=dtype)

    # Initialize V and H from the seed.
    m = r_keep
    for j in range(r_keep):
        w = Jv(V[:, j])
        for i in range(m):
            hij = np.vdot(V[:, i], w); H[i, j] = hij; w -= hij * V[:, i]
        hj = np.linalg.norm(w); H[m, j] = hj

        # Append the new vector w to V
        if hj > 0.0 and m < m_target:
            V[:, m] = w / hj
            m += 1
        else:
            LOG.verbose(f'Happy breakdown at index {m} because {hj} and {m_target}')

    # Start Arnoldi scheme from V - the updated seed.
    while m < m_target:
        j = m - 1
        w = Jv(V[:, j])
        for i in range(m):
            hij = np.vdot(V[:, i], w); H[i, j] = hij; w -= hij * V[:, i]
        hj = np.linalg.norm(w); H[m, j] = hj

        if hj == 0.0: break # Happy breakdown
        V[:, m] = w / hj
        m += 1

    # Return the orthonormal vectors and the associated upper-Hessenberg matrix.
    return V[:,:m], H[:m, :m]

def _pick_near_axis(vals: np.ndarray, 
                    r: int, 
                    omega_min: float) -> np.ndarray:
    """Helper function to pick the eigenvalue closest to the imaginary axis that
    has a non-zero imaginary component.
    
    Parameters
    ----------
    vals : ndarray
        The current eigenvalues.
    r : int
        The number of eigenvalues to maintain.
    omega_min : float
        Minimal threshold to select eigenvalues that have a non-zero imaginary component.

    Returns
    -------
    minimal_eigvals : ndarray
        (Complex conjugated pair of) eigenvalues closest to the imaginary axis.
    """
    mask = np.where(np.abs(np.imag(vals)) > omega_min)[0]
    if mask.size == 0:
        mask = np.arange(vals.size)
    return mask[np.argsort(np.abs(np.real(vals[mask])))[:min(r, mask.size)]]

def initializeHopf(G: Callable[[np.ndarray, float], np.ndarray],
                   u : np.ndarray,
                   p : float,
                   m0: int,
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
    m0 : int
        The initial number of eigenpairs to compute.
    sp : Dict
        Solver parameters including arguments `keep_r` and `m_target`.

    Returns
    -------
    state: Dict
        Contains the current eigenspace "V", the Hessenberg matrix "H", the
        Ritz values and vectors "ritz_vals" and "ritz_vecs", index "lead" of the eigenvalue 
        closes to the imaginary axis, and "omega" the imaginary part of this eigenvalue.
    """
    M = len(u)
    rdiff = sp["rdiff"]
    Jv = lambda v: (G(u + rdiff*v, p) - G(u - rdiff*v, p)) / (2.0*rdiff)

    # Compute the initial seed of many eigenvectors with largest real part (see assumption).
    if M > 2:
        k_pool = min(m0, max(1, M-2))
        A = slg.LinearOperator((M, M), Jv)
        vals_full, V = slg.eigs(A, k=k_pool, which="LR", return_eigenvectors=True) # type: ignore[reportAssignmentType]

        # Compute Ritz eigenvalues and vectors (vals and vecs)
        Q, R = np.linalg.qr(V)  # Q: (n,k), R: (k,k)
        Rinv = np.linalg.solve(R, np.eye(R.shape[0]))
        H = R @ np.diag(vals_full) @ Rinv
        ritz_vals, ritz_vecs = np.linalg.eig(H)
        state = {"V": Q, "H": H, "ritz_vals": ritz_vals, "ritz_vecs": ritz_vecs}
    elif M == 2: # edge case M = 2. Compute Ritz values explicitly
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.0, 1.0])
        J  = np.column_stack((Jv(e1), Jv(e2)))

        V = np.eye(2)             # orthonormal basis spanning R^2
        H = J                     # H = V^* J V = J
        ritz_vals, ritz_vecs = np.linalg.eig(H)
        state = {"V": V, "H": H, "ritz_vals": ritz_vals, "ritz_vecs": ritz_vecs}

    # Do one refresh step to maintain a consistent data structure and return.
    LOG.verbose(f"Initializing Hopf")
    return refreshHopf(G, u, p, state, sp)

def refreshHopf(G: Callable[[np.ndarray, float], np.ndarray],
                u : np.ndarray,
                p : float,
                state: Dict,
                sp: Dict) -> Dict:
    """
    Update the Hopf bifurcation detection function at the new point starting from
    information at the previous point.

    Parameters
    ----------
    G : Callable
        The objective function.
    u : ndarray
        The current state vector on the path.
    p : float
        The current parameter value.
    state: Dict
        The Hopf eigenstate at a previous point, used as seed for the new Arnoldi method.
    sp : Dict
        Solver parameters including arguments `keep_r` and `m_target`.

    Returns
    -------
    state: Dict
        Contains the current eigenspace "V", the Hessenberg matrix "H", the
        Ritz values and vectors "ritz_vals" and "ritz_vecs", index "lead" of the eigenvalue 
        closes to the imaginary axis, and "omega" the imaginary part of this eigenvalue.
    """
    rdiff = sp["rdiff"]
    keep_r = sp["r_keep"]
    m_target = sp["m_target"]
    omega_min = 1e-3
    Jv = lambda v: (G(u + rdiff*v, p) - G(u - rdiff*v, p)) / (2.0*rdiff)

    # Pick the Ritz pair closest to the imaginary axis and lift them to the full-dimensional space
    idx = _pick_near_axis(state["ritz_vals"], keep_r, omega_min)
    V0  = state["V"] @ state["ritz_vecs"][:, idx]   # lift reduced Ritz vecs to full space
    V0  = _orthonormalize(V0)

    # Update the eigenmodes starting from V0 as seed
    V, H = arnoldi_from_seed(Jv, V0, m_target)
    ritz_vals, ritz_vecs = np.linalg.eig(H)

    # Pick the current leading complex conjugated eigenpair and return a state dict.
    lead = int(_pick_near_axis(ritz_vals, 1, omega_min)[0])
    omega=float(abs(np.imag(ritz_vals[lead])))
    state = {"V": V, "H": H, "ritz_vals": ritz_vals, "ritz_vecs": ritz_vecs, "lead": lead, "omega": omega}
    LOG.verbose(f"Lead Eigenvalue {ritz_vals}")
    return state

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
    prev_leading_ritz_value = prev_state["ritz_vals"][prev_state["lead"]]
    curr_leading_ritz_value = curr_state["ritz_vals"][curr_state["lead"]]

    return np.real(prev_leading_ritz_value) * np.real(curr_leading_ritz_value) < 0.0