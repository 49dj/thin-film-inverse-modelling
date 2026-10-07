# Units, objectives and source-preserving changes

Source modules: Q2.py (SiC) and Q3.py (Si) from the modelling project's supporting
code. Only functions and physical constants were extracted. Original data-loading
defaults, spreadsheets, machine-specific output paths, interactive plotting
startup, global warning suppression and BLAS environment changes are not imported.

## SiC / Q2

Wavenumbers and phonon dampings: cm^-1. Carrier concentrations: cm^-3. Mobility:
cm²/(V s). Film thickness parameter: micrometres, converted to metres in the
Fresnel function. The original bounded nine-parameter fit and optional LOPC term
are preserved. LS uses soft-L1 with f_scale=0.01; DE uses squared residuals.

Known legacy caveat: `fit_stack_enhanced(method='hybrid')` compares robust LS cost
against DE SSE even though the scales/objectives differ. The source also returns
the initial point on optimiser failure and can retain infinite best_cost. Those
behaviours remain visible rather than being presented as a validated selection
rule. `joint_fod_weight` is reported metadata but does not enter the residual.
The demo does not exercise this legacy hybrid selection.

## Si / Q3

Film thickness: µm; oxide thickness: nm; wavenumbers: cm^-1. Q3 has 19 shared
parameters and one reflectance scale per angle. Bounds are retained literally
from `bounds_and_init`. The film has an empirical high-frequency real-permittivity
correction; the substrate does not receive that correction. Angular, film and
oxide-thickness averaging are discrete weighted Gaussian samples.

The residual concatenates `(model-observation)*w`, where w=4 between 700–1100
cm^-1, w=2 from 1500 cm^-1 upwards, and w=1 elsewhere. Consequently SSE uses
squared weights 16 and 4. Ten identical entries `0.1*(d-d0)` encode the original
thickness penalty. DE minimises this SSE on a subsampled grid with coarser averaging;
LS minimises the full-grid soft-L1 loss with f_scale=0.03. No common objective value
is used to claim algorithm superiority.

`fit_silicon` adds an explicit seed (42; original Q3 DE had no seed), one worker,
and configurable budgets. The iteration-limited smoke check disables DE polish;
the public full-fit default retains it. Equations, bounds and loss parameters are unchanged.
It requires explicit `d0`; the original FFT initializer is not advertised because
its frequency/thickness conversion is not independently confirmed. Jacobian-based
confidence intervals from the source are not exposed as calibrated uncertainty.

## Synthetic demonstration

The reduced default fit is newly written orchestration around the unchanged Q3
forward model. It fixes nuisance parameters to independently chosen defaults,
uses true thickness 5.4 µm and searches [3,8] µm on two 10°/15° synthetic spectra.
It omits the original thickness penalty to demonstrate known-parameter recovery.
The separate `--full-fit` flag restores the original multi-parameter Q3 objective.
Neither example uses experimental data or reproduces a manuscript figure.
