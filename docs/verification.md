# Verification, 7 October 2026

Tested with Python 3.12.10 on Windows; direct dependency versions are pinned in
`requirements.txt`. Editable package installation and `pip check` passed.

- `python -m pytest -q`: 3 tests passed, including an iteration-limited full Q3
  objective DE/LS execution with polish disabled (about 10 seconds).
- `python scripts/synthetic_demo.py`: completed. Known synthetic thickness
  5.4 µm; estimated 5.400000000000001 µm; residual RMSE about 2.14e-17.
- Default demo DE exhausted its iteration budget (`de_success=false`); the
  subsequent least-squares step reported success. This status is preserved in JSON.

The near-zero noiseless residual is a self-consistency example using fixed known
nuisance parameters, not empirical validation. Full-budget multi-parameter Q3
convergence and full legacy Q2 fitting were not established. No experimental
dataset or paper figure was reproduced.

`CITATION.cff` passed validation against the official CFF 1.2 schema.
