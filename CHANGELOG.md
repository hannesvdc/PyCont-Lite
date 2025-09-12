# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Trying tangent computation using Newton-Schur method** but L-GMRES still takes 10 iterations

### Changed
- **Computing tangent using bordered system** for better conditioning and improved stability
- **Limit L-GMRES Iterations to min(M, 10)** for faster tangent computations without limiting accuracy.

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