# Figure regeneration

The chemical-subgroup figure is generated from the public 96-row RMSE table:

```bash
python scripts/render_chemical_subgroup_figure.py
```

The command creates PDF, PNG, and fully editable PPTX versions under `figures/generated/`. It requires the packages pinned in `environment/analysis-requirements.txt`.

The generator does not outline the four subgroup rows with fewer than 100 molecules. Their `sparse_n_lt_100` status and exact sample sizes remain available in the source CSV and Supplementary Table S17.

Version 1.0.4 widens the space between chemical-axis names and subgroup labels to match the revised manuscript. The 192 heatmap cells, evaluation order, colour scale, and numerical inputs are unchanged. The PDF/PNG generator checks label separation and horizontal bounds before saving; the editable PPTX uses the same widened columns.
