# Figure regeneration

The chemical-subgroup figure is generated from the public 96-row RMSE table:

```bash
python scripts/render_chemical_subgroup_figure.py
```

The command creates PDF, PNG, and fully editable PPTX versions under `figures/generated/`. It requires the packages pinned in `environment/analysis-requirements.txt`.

The generator does not outline the four cells with fewer than 100 molecules. Their `sparse_n_lt_100` status and exact sample sizes remain available in the source CSV and Supplementary Table S13.
