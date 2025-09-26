import numpy as np

import pycont

def normalHopf():
    def G(u : np.ndarray, mu : float) -> np.ndarray:
        x = u[0]; y = u[1]
        Gx = mu*x - y - (x**2 + y**2) * x
        Gy = x + mu*y - (x**2 + y**2) * y
        return np.array([Gx, Gy])
    mu0 = -1.0
    u0 = np.array([0.0, 0.0])

    ds_max = 0.01
    ds_min = 1.e-6
    ds = 0.01
    n_steps = 200
    solver_parameters = {"tolerance": 1e-10}
    continuation_result = pycont.arclengthContinuation(G, u0, mu0, ds_min, ds_max, ds, n_steps, solver_parameters, 'verbose')

    pycont.plotBifurcationDiagram(continuation_result, p_label=r'$\mu$')

if __name__ == '__main__':
    normalHopf()