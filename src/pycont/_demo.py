import numpy as np
from pycont import arclengthContinuation, Verbosity

def main():
    G = lambda u, p: p*u - u**3
    u0 = np.array([-3.0])
    p0 = 9.0
    arclengthContinuation(
        G, u0, p0,
        ds_min=1e-3, ds_max=1e-1, ds_0=1e-2, n_steps=20,
        solver_parameters={"nk_maxiter": 10, "tolerance": 1e-10},
        verbosity=Verbosity.INFO,
    )

if __name__ == "__main__":
    main()