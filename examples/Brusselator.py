import numpy as np
import pycont

def Brusselator():
    A = 1.0
    G = lambda x, B : np.array([A - (B+1.0)*x[0] + x[0]**2 * x[1], B*x[0] - x[0]**2 * x[1]])
    B0 = 1.0
    x0 = np.array([A, B0/A])

    tolerance = 1e-10
    ds_max = 0.01
    ds_min = 1e-6
    ds0 = 1e-3
    n_steps = 1000
    solver_parameters = {"tolerance" : tolerance, "hopf_detection" : True}
    continuation_result = pycont.arclengthContinuation(G, x0, B0, ds_min, ds_max, ds0, n_steps, solver_parameters)

    pycont.plotBifurcationDiagram(continuation_result)

if __name__ == '__main__':
    Brusselator()