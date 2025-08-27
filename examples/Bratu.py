import numpy as np
import matplotlib.pyplot as plt

import pycont

def BratuTest():
    """
    Bratu: u_xx + lambda * exp(u) = 0 on (0,1) with Dirichlet boundary conditions.
    Plots the bifurcation diagram with a fold point near lambda = 3.5.
    """

    N = 101 # total number of points
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0]
    M = N - 2

    def G(u: np.ndarray, lam: float) -> np.ndarray:
        u_full = np.zeros(N, dtype=float)
        u_full[1:-1] = u
        
        u_xx = (u_full[:-2] - 2.0 * u_full[1:-1] + u_full[2:]) / (dx * dx)
        r = u_xx + lam * np.exp(u_full[1:-1])
        return r
    
    # We know that u = 0 for lambda = 0 - otherwise we must solve G(u, r0) = 0.
    lam0 = 0.0
    u0 = np.zeros(M)

    # Do continuation
    ds_max = 0.01
    ds_min = 1e-6
    ds0 = 1e-4
    n_steps = 2000
    solver_parameters = {"tolerance": 1e-10, "bifurcation_detection": False}
    continuation_result = pycont.pseudoArclengthContinuation(G, u0, lam0, ds_min, ds_max, ds0, n_steps, solver_parameters=solver_parameters)

    # Print some Internal info
    print('\nNumber of Branches:', len(continuation_result.branches))
    print('Bifurcation Points:', continuation_result.bifurcation_points)

    # Plot the bifurcation diagram (lambda, max(u))
    for branch in continuation_result.branches:
        max_u = np.sign(branch["u"][:, 50]) * np.max(np.abs(branch["u"]), axis=1)
        plt.plot(branch["p"], max_u, color="tab:blue")
    plt.plot(lam0, np.max(u0), 'go', label='SP')
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$u(x)$")
    plt.show()

if __name__ == '__main__':
    BratuTest()