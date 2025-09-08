# PyCont-Lite

[![PyPI version](https://badge.fury.io/py/pycont-lite.svg)](https://badge.fury.io/py/pycont-lite)

**PyCont-Lite** is a lightweight, matrix-free arclength continuation library for solving nonlinear parametric systems

$$
G(u, p) = 0
$$

with **automatic bifurcation detection, branch switching, and stability analysis**. 

## Key Features
- **Matrix-free**: only need to implement $G(u, p)$, no Jacobians required.  
- **Adaptive continuation**: robust predictor–corrector with adjustable step sizes.  
- **Bifurcation detection**: automatically localizes and classifies folds and branch points.  
- **Branch switching**: continues all discovered branches recursively.  
- **Stability analysis**: computes leading eigenvalues to distinguish stable vs. unstable segments.  
- **Lightweight plotting**: quickly visualize bifurcation diagrams with `plotBifurcationDiagram`. Plots automatically distinguish stable vs. unstable branches.
- **Structured output**: branches (with stability) and events (folds, bifurcations) are returned for further analysis.”

## Installation & Requirements
PyCont-Lite requires only:
- `numpy`
- `scipy`
- `matplotlib`

Install via PyPI (pip):
```
pip install pycont-lite
```

## Quick Examples

### Pitchfork Bifurcation

```python
import numpy as np
import pycont

# Define the pitchfork function
def G(u, p):
    return u**3 - p*u

# Initial guess
u0 = np.array([-3.0])
p0 = 0.0

# Run continuation
ds_max = 0.01
ds_min = 1.e-6
ds = 0.001
n_steps = 1000
solver_parameters = {"tolerance": 1e-10}
continuation_result = pycont.arclengthContinuation(G, u0, p0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

# Plot the solution curve
pycont.plotBifurcationDiagram(continuation_result)
```

<p align="center">
    <img src="https://raw.githubusercontent.com/hannesvdc/PyCont-Lite/main/docs/images/Pitchfork.png" width="400">
</p>

### Fold Bifurcation

```python
import numpy as np
import pycont

# Define the fold function
G = lambda x, r: r + x**2

# Initial guess
u0 = np.array([-5.0])
p0 = -u0[0]**2

# Run continuation
ds_max = 0.01
ds_min = 1.e-6
ds = 0.001
n_steps = 5000
solver_parameters = {"tolerance": 1e-10}
continuation_result = pycont.arclengthContinuation(G, u0, p0, ds_min, ds_max, ds, n_steps, solver_parameters=solver_parameters)

# Plot the curves
pycont.plotBifurcationDiagram(continuation_result)
```
<p align="center">
    <img src="https://raw.githubusercontent.com/hannesvdc/PyCont-Lite/main/docs/images/Fold.png" width="400">
</p>

### Bratu: PDE Example

The nonlinear boundary value problem

$$
u_{xx} + \lambda e^u = 0, \quad u(0)=u(1)=0
$$

discretized with finite differences:

```python
import numpy as np
import pycont

N = 101 # total number of points
x = np.linspace(0.0, 1.0, N)
dx = x[1] - x[0]

def G(u: np.ndarray, lam: float) -> np.ndarray:
    u_full = np.zeros(N, dtype=float)
    u_full[1:-1] = u
    
    u_xx = (u_full[:-2] - 2.0 * u_full[1:-1] + u_full[2:]) / (dx * dx)
    r = u_xx + lam * np.exp(u_full[1:-1])
    return r

# We know that u = 0 for lambda = 0 - otherwise we must solve G(u, lambda0) = 0.
lam0 = 0.0
u0 = np.zeros(N-2)

# Do continuation
ds_max = 0.01
ds_min = 1e-6
ds0 = 1e-4
n_steps = 2000
solver_parameters = {"tolerance": 1e-10}
continuation_result = pycont.arclengthContinuation(G, u0, lam0, ds_min, ds_max, ds0, n_steps, solver_parameters=solver_parameters)

# Plot the bifurcation diagram (lambda, max(u))
u_transform = lambda u : np.sign(u[50]) * np.max(np.abs(u))
pycont.plotBifurcationDiagram(continuation_result, u_transform=u_transform, p_label=r'$\lambda$', u_label=r'$\text{sign}(u) ||u||_{\infty}$')
```

This produces the classical S-shaped bifurcation curve with a fold near $\lambda \approx 3.51$.

<p align="center">
    <img src="https://raw.githubusercontent.com/hannesvdc/PyCont-Lite/main/docs/images/Bratu.png" width="400">
</p>

## Plotting Bifurcation Diagrams

PyCont-Lite includes a helper function `plotBifurcationDiagram` for quick visualization. Stable segments are shown as solid lines and unstable segments as dashed lines, just like in AUTO/MATCONT.
By default, it plots the parameter value on the horizontal axis and the transformed state variable on the vertical axis.

```python
import pycont

# After running continuation...
pycont.plotBifurcationDiagram(continuation_result)
```

For multi-dimensional systems, you can specify how to reduce $u$ to a single scalar for plotting via the `u_transform` argument.
 - Default behavior:
	- If $u$ has dimension 1 → plot $u$ directly.
	- If $u$ has dimension >1 → plot the Euclidean norm $||u||$.
 - Custom transform (example: plot the maximum component of $u$):
```python
pycont.plotBifurcationDiagram(result, u_transform=lambda u: u.max())
```

## Solver Parameters

You can fine-tune the solver by passing a dictionary:
```python
solver_parameters = {
    "rdiff": 1e-8,              # finite-difference step
    "nk_maxiter": 20,           # Newton-Krylov iterations
    "tolerance": 1e-10,         # nonlinear tolerance
    "bifurcation_detection": True,
    "analyze_stability": True,  # compute leading eigenvalue
    "initial_directions": "both"  # 'both', 'increase_p', 'decrease_p'
}
```

## Output Format

```arclengthContinuation``` returns a **ContinuationResult** object with:
 - **branches**: list of Branch objects (u_path, p_path, stability flag, etc.)
 - **events**: list of Event objects (start points, folds, bifurcations)

This makes it easy to explore and plot bifurcation diagrams programmatically.

## Planned Features
The following features are under active consideration for future releases:
- Hopf bifurcation detection
- Limit cycle continuation
- Improved tangent computation for high-dimensional PDE systems

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Acknowledgement

I started this project because most continuation software is either legacy Fortran (AUTO), Matlab-only (MATCONT, COCO), or heavyweight.
PyCont-Lite is meant to be modern, lightweight, and useful both for industry and academia!

For feature requests or contributions, feel free to open an issue or reach out: hannesvdc[at]gmail[dot]com.