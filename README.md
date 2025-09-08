# PyCont-Lite

[![PyPI version](https://badge.fury.io/py/pycont-lite.svg)](https://badge.fury.io/py/pycont-lite)

**PyCont-Lite** is a simple, matrix-free arclength continuation library for solving parametric nonlinear systems of the form

$$
G(u, p) = 0
$$

with automatic detection of bifurcation points and branch switching. In particular, PyCont uses

- Matrix-free implementation through nonlinear Newton-Krylov solvers
- Precise bifurcation point localization (via bisection)
- Branch switching at bifurcation points
- Lightweight and easy to integrate into your workflows
- No derivatives or Jacobians needed! Everything works directly on the level of G(u, p).

Installation
---
```
pip install pycont-lite
```

Minimal Working Example (Pitchfork Bifurcation)
---
<pre>
import numpy as np
import pycont

# Define the pitchfork function
def G(u, p):
    return u**3 - p*u

# Initial guess
u0 = np.array([2.0])
p0 = 0.0

# Run continuation
continuation_result = pycont.pseudoArclengthContinuation(G, u0, p0, ds_min=0.001, ds_max=0.05, ds=0.01, n_steps=200)

# Plot the solution curve
pycont.plotBifurcationDiagram(continuation_result)
</pre>


License
---
This project is licensed under the MIT License — see the LICENSE file for details.

Acknowledgement
--- 
I started this project because there are (in my opinion) no good, lightweight, and easy to use arclength continuation packages 
in the python ecosystem. These days, researchers are too often forced to use ancient Fortran packages like AUTO, or language-specific 
tools like MATCONT or COCO. PyCont-Lite seeks to create an elegent package with all features for Python - the default language for
scientific computing and machine learning. 

If you want to see new features, either contact me or write a pull request. I will continue updating PyCont-Lite based on my own needs, and soon
PyCont will include Hopf bifurcation detection and limit cycle detection. For questions or requests, feel free to contact me at hannesvdc[at]gmail[dot]com. 
