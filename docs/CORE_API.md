# Canonical evaluation and blending core

The first canonical code-export gate contains four small modules under `src/dlg_sol/`:

- `evaluations.py` loads and validates the six-evaluation registry in `configs/evaluations.json`;
- `label_firewall.py` separates label-free prediction assembly from label-bearing scoring and rejects scored-label use for fitting operations;
- `blend.py` fits or applies one scalar convex coefficient with the neural prediction weighted by `alpha`;
- `metrics.py` computes RMSE, MAE, prediction-minus-observation bias, and paired molecular-row bootstrap intervals for an RMSE difference.

The fixed blend is

```text
prediction = (1 - alpha) * descriptor_prediction + alpha * neural_prediction
```

`fit_convex_weight` performs no-intercept least squares for `observed - descriptor` against `neural - descriptor`, clips the coefficient to `[0, 1]`, and refuses labels marked as belonging to scored rows. The caller must explicitly state the label partition. AqSolDBc uses the prespecified coefficient recorded in the evaluation registry and therefore does not call the fitting function for its primary result.

`paired_rmse_bootstrap` also requires an explicit seed because the recorded analysis families used different prespecified seeds. The function does not silently substitute one global seed.

Run the core contract tests from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The fixture in `tests/fixtures/` is synthetic and contains no benchmark molecules or experimental measurements. `results/verified_manifests/core_module_parity.json` records an internal full-row parity check without redistributing the row-level inputs.

The current core does not train the descriptor, language, geometry, or fusion models. Those branches remain separate later export gates.
