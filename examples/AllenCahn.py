import numpy as np
import pycont

def laplace_neumann(u, dx):
    u_ext = np.hstack([u[1], u, u[-2]])  # reflective ghosts
    u_l = np.roll(u_ext, -1)[1:-1]
    u_r = np.roll(u_ext,  1)[1:-1]
    return (u_l - 2.0*u + u_r) / dx**2

def AllenCahnTest():
    """
    Allen-Cahn PDE
        eps phi_xx - W'(phi) / eps = 0
    with bimodal W(phi) = (phi^2 - 1)^2 / 4 and homogenenous Neumann boundary conditions. 
    `eps` is the continuation parameter.
    """
    N = 100 # total number of points for u and v
    x = np.linspace(-1.0, 1.0, N)
    dx = (x[-1] - x[0]) / (N-1)

    # Build the Allen-Cahn objective function through finite differences
    def G(phi : np.ndarray, eps : float):
        phi_xx = laplace_neumann(phi, dx)
        rhs = eps * phi_xx - phi * (phi**2 - 1.0) / eps
        return rhs
    
    # We use an initial point on the default branch phi(x) = 0.0
    eps0 = 0.6
    phi0 = np.zeros(N)

    # Do continuation
    tolerance = 1e-9
    ds_max = 1e-3
    ds_min = 1e-6
    ds0 = 1e-4
    n_steps = 1000
    solver_parameters = {"tolerance" : tolerance, "param_min" : 0.22, "param_max" : 0.7}
    continuation_result = pycont.arclengthContinuation(G, phi0, eps0, ds_min, ds_max, ds0, n_steps, solver_parameters=solver_parameters)

    # Plot the bifurcation diagram eps versus phi(x=-1)
    u_transform = lambda phi: phi[0]
    pycont.plotBifurcationDiagram(continuation_result, u_transform=u_transform, p_label=r'$\varepsilon$', u_label=r'$\phi(x=-1)$')

if __name__ == '__main__':
    AllenCahnTest()