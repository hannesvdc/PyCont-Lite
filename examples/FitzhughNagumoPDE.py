import pathlib

import numpy as np
import scipy.optimize as opt

import pycont

def sigmoid(x_array, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x_array  - x_center)/x_scale)) + y_center

def laplace_neumann(u, dx):
    u_ext = np.hstack([u[1], u, u[-2]])  # reflective ghosts
    u_l = np.roll(u_ext, -1)[1:-1]
    u_r = np.roll(u_ext,  1)[1:-1]
    return (u_l - 2.0*u + u_r) / dx**2

def FitzhughNagumoTest():
    """
    Fitzhugh-Nagumo PDEs:
        u_xx + u - u^3 - v = 0
        delta v_xx + eps*(u - a1v - a0) = 0
    with homogenenous Neumann boundary conditions. `eps` is the continuation parameter.

    This model is a modification of the FHN ODEs.
    """
    N = 100 # total number of points for u and v
    L = 20.0 # Size of the domain
    x = np.linspace(0.0, L, N)
    dx = L / (N-1)

    # Build the FHN objective function through finite differences
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    def G(z : np.ndarray, eps : float):
        u, v = z[:N], z[N:]
        u_xx = laplace_neumann(u, dx)
        v_xx = laplace_neumann(v, dx)
        u_rhs = u_xx + u - u**3 - v
        v_rhs = delta * v_xx + eps * (u - a1*v - a0)
        return np.concatenate((u_rhs, v_rhs))
    
    # Load the initial condition from file - otherwise compute it
    eps0 = 0.1
    here = pathlib.Path(__file__).parent
    datafile = here / "data" / "fhn_init.npy"
    if datafile.exists():
        z0 = np.load(datafile)
    else:
        # We know the initial condition looks like a sigmoid
        u0 = sigmoid(x, 14.0, -1, 1.0, 2.0)
        v0 = sigmoid(x, 15, 0.0, 2.0, 0.1)
        z0 = opt.newton_krylov(lambda z : G(z, eps0), np.concatenate((u0, v0)), f_tol=1e-9, method='lgmres')
        np.save(datafile, z0)
    print('Initial FHN Residual:', np.linalg.norm(G(z0, eps0)))

    # Do continuation. The Fitzhugh-Nagumo equations are much worse conditioned than the Bratu PDE because
    # the Jacbian is very non-normal. We must use a small tolerance.
    tolerance = 1e-10
    ds_max = 0.01
    ds_min = 1e-6
    ds0 = 1e-3
    n_steps = 1000
    solver_parameters = {"tolerance" : tolerance, "param_min" : 0.01, "hopf_detection" : True}
    continuation_result = pycont.arclengthContinuation(G, z0, eps0, ds_min, ds_max, ds0, n_steps, solver_parameters, verbosity='verbose')

    # Plot the bifurcation diagram eps versus <u>
    u_transform = lambda z: np.average(z[:N])
    pycont.plotBifurcationDiagram(continuation_result, u_transform=u_transform, p_label=r'$\varepsilon$', u_label=r'$<u>$')
    
if __name__ == '__main__':
    FitzhughNagumoTest()