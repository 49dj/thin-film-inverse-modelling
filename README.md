# Thin-film inverse modelling

Optical reflectance encodes film thickness and material dispersion through an
oscillatory, often non-identifiable inverse problem. This repository collects the
coherent numerical core of a collaborative mathematical-modelling project on
SiC and Si spectra. Yizhe Hu participated in the project; individual ownership of
every source component has not been independently established.

## Methods

- SiC: phonon/free-carrier dielectric response, complex Fresnel coefficients,
  oblique unpolarised thin-film interference, bounded nonlinear fitting.
- Si: Drude silicon and Lorentz oxide dispersion, air/oxide/film/substrate Abeles
  transfer matrices, angular/thickness averaging, shared dual-angle parameters.
- Original Si objective: weighted spectral residuals plus a thickness penalty;
  coarse differential evolution SSE followed by full-grid soft-L1 least squares.
- Residual plots and a compact, independently generated dual-angle demonstration.

The SiC and Si models remain separate; their materials, units and objectives are
not interchangeable. See [methods and retained ambiguities](docs/methods.md).

## Run

Python 3.12; commands run from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/synthetic_demo.py
```

The default demonstration estimates only film thickness from two noiseless
synthetic Si spectra while keeping the other known generator parameters fixed.
Its objective has no prior thickness penalty. This is an explicitly simplified
software demonstration, not a reproduction of the competition analysis or proof
of identifiability in real experiments. Outputs: `results/fit.json` and residual
figure `results/dual_angle.png`.

```bash
python scripts/synthetic_demo.py --full-fit
```

`--full-fit` selects the original Q3 multi-parameter objective through the new
wrapper; it is expensive and convergence is not guaranteed. The original Q2
SiC `fit_stack_enhanced` API is available in `thinfilm.sic_fitting`, including its
documented legacy caveats. The full scientific budgets are not exercised by the
lightweight tests.

## Structure and checks

`src/thinfilm/forward_model.py` retains SiC optics; `silicon.py` retains Si/oxide
optics; `objective.py` retains Q3 curves, residuals and bounds; `optimisation.py`
exposes Q3 DE/LS; `sic_fitting.py` preserves Q2 fitting. `diagnostics.py` and
`plotting.py` report residuals. `examples/` and `tests/` use only generated spectra.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check zero-thickness/direct-interface consistency, exact generated Q3
residuals, bounds and finite execution of an iteration-limited full-objective fit.
They do not validate experimental accuracy, uncertainty calibration or superiority
of one optimiser. Source competition spreadsheets, reports, exploratory neural
and Bayesian methods, and internal documents are excluded.

## Limits and attribution

Spectral weights, dispersion corrections, thickness penalties and bounds are
source assumptions, not universal material laws. Parameters can compensate for
one another. Reflectance clipping in the retained Q3 model can conceal physically
questionable intermediate values. No PINN, Pyro, calibrated intervals or experimental
benchmark is claimed. Cite the repository URL and commit; no verified publication
DOI for this project is asserted. [Rights and collaboration](docs/rights.md).

See [observed verification results](docs/verification.md) for executed checks and
their limits.
