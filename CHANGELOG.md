# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
- **Fast Hopf Updates** using the Jacobi-Davidson algorithm.
- **Accurate Hopf localization** using the Jacobi-Davidson algorithm.

### Changed
- **Hysteresis for Hopf detection** after restarting from a prior Hopf point.

## [0.5.0] - 2025-10-06

### Added
- **High-Precision Hopf Localization** after Hopf detection by means of precise eigenvalue computations.

### Changed
- **Created Detection Modules** for param min, param max, fold, bifurcation and Hopf detection. Uses same underlying numerical code and algorithms, just gives for a cleaner continuation structure that loops over the detection modules.

### Removed
- **Bifurcation.py** is now in detection/_bifurcation.py
- **Hopf.py** is now in detection/_hopf.py
- **Fold Localization** is now in detection/_fold.py
- **Param_min / Param_max** moved from ArclengthContinuation.py to detection/parammin.py and detection/parammax.py respectively.

## [0.4.0] - 2025-09-28

### Added
- **Orthonormalizing** l- and r- vectors for bifurcation detection. Also projecting all l-vectors off the initial tangent.
- **Multiple Bifurcation Test vectors** for more robust bifurcation detection.
- **Hopf Bifurcation Detection** by keeping track of the few rightmost eigenvalues and updating with a Rayleigh iteration.
- **Readme** now contains the normal Hopf example and the Hopf point in the Fitzhugh-Nagumo PDEs.

### Changed
- **Bifurcation detection before fold detection** because BP has priority.
- **Internal representations** of bifurcation detection states in advance of DetectionModules.

### Removed
- **testfunctions.py** the bordered test for bifurcation points is already included in Bifurcation.py

## [0.3.1] - 2025-09-19

### Added
- **Demo script** to test installation
- **Readme** now contains the Allen-Cahn example and how to use Verbosity.

## [0.3.0] - 2025-09-18

### Added
- **Trying tangent computation using Newton-Schur method** but L-GMRES still takes 10 iterations
- **Fitzhugh-Nagumo PDE** new example of a hard PDE that has a fold point and a Hopf point (to detect later).
- **tolerance = max(a_tol, r_diff)** because we can never theoretically go below r_diff.
- **Plotting DSFLOOR** event to provide more information.
- **Arclength information in Branches and Events** for better plotting and debugging.
- **BrentQ-based optimizer** for fold and bifurcation point localization.
- **Newton-Krylov corrector** for LGMRES in tangent computation, test function evaluation and bifurcation point detection algorithms.
- **Logger class** that handles the degrees of verbosity. Options are 'off', 'info' (default), and 'verbose'. Instance shared by all modules.
- **User input checks** that raises InputErrors when initial values and parameters don't make sense or contains NaNs / Infs.

### Moved
- **Test function and BP Localizer** to new Bifurcation.py
- **Fold point localizer** to Tangent.py

### Changed
- **Computing tangent using bordered system** for better conditioning and improved stability
- **Limit L-GMRES Iterations to min(M, 10)** for faster tangent computations without limiting accuracy.
- **Replaced ds increase/decrease logic** after Newton-Krylov solver by looking at the residual norm, not relying on scipy's crude info. Works on all examples.
- **Branch creation and addig points**: reusing logic by function calls instead of copying code.
- **Replaced Bifurcation Detection** with a stable sign-change detector based on the Jacobian of the extended objective function.

### Removed
- **Slow bisection-based localizer** for folds and bifurcation points.

## [0.2.0] - 2025-09-08
### Added
- **First multidimensional example**: Bratu problem (101-point discretization) now works out of the box.
- **Robust bifurcation detection**:
  - Improved bordered system formulation with regularization.
  - Guards against Krylov solver blow-ups using `np.errstate` and convergence checks.
  - Cleaner numerical bisection routines for localization.
- **Improved fold detection**:
  - Explicit fold localization via tangent component sign check.
  - Optional classification of fold events vs bifurcation points.
- **Solver parameters system** (`solver_parameters` dict) with multiple configurable options:
  - `rdiff` – finite-difference step size.
  - `nk_maxiter` – maximum Newton–Krylov iterations.
  - `tolerance` – nonlinear residual tolerance.
  - `bifurcation_detection` – enable/disable test functions for speed.
  - `analyze_stability` – compute leading eigenvalue to mark branches stable/unstable.
  - `initial_directions` – choose `'both'`, `'increase_p'`, or `'decrease_p'`.
- **Plotting tool**: `plotBifurcationDiagram` for quick visualization of branches and events.
- **Standardized output**: continuation now returns `Branch` and `Event` objects via `ContinuationResult`.
- **Better tangent initialization**:
  - Initial tangent computed via secant/Newton–Krylov hybrid.
  - More robust fallback strategies for noisy problems.
- **Documentation**:
  - Added docstrings for all public and internal functions.
  - Clarified parameter roles in continuation and branch switching.

### Changed
- **Event/Branch structure**:
  - Branches carry `from_event` and `termination_event`.
  - Events are shared across branches (no duplicated event IDs).
- **Fold handling**: continuation now stops at fold points (segment mode), but can resume if desired.
- **Stability analysis**: stability is evaluated once per branch (not per step) for efficiency.
- **Cleaner caching**: directional derivative operators (`Gu_v`, extended systems) streamlined for readability.
- **Renamed** main continuation function from 'pseudoArclengthContinuation' to 'arclengthContinuation'.

### Fixed
- Fixed spurious `RuntimeWarning`s in Krylov solvers during bifurcation detection.
- Fixed dimension issues when plotting scalar vs vector-valued branches (`u_transform`).
- Fixed numerical errors in one-dimensional test cases (pitchfork, transcritical, fold).
- Fixed event duplication by checking uniqueness tolerance.


## [0.1.0] - 2025-07
### Added
- Initial public release of **PyCont-Lite**.
- Pseudo-arclength continuation with adaptive step control.
- Support for fold, pitchfork, and transcritical test cases.
- Matrix-free Newton–Krylov solver backend.
- Basic bifurcation detection via bordered system test function.
- Minimal examples and tests.
