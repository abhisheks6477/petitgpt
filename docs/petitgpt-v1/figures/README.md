# Technical-report figures

Run from the repository root with Python and Matplotlib installed:

```bash
python docs/petitgpt-v1/figures/plot_report_figures.py
```

The script reads only the companion public CSVs and writes PNG and SVG versions
beside itself. It needs no model, GPU, network access or private data. The checked-in
figures were rendered with Matplotlib 3.10.0; other versions may change typography.

| Figure | Input | Interpretation |
|---|---|---|
| `pretraining_validation` | [PRETRAIN_VALIDATION_CURVE.csv](../tables/PRETRAIN_VALIDATION_CURVE.csv) | All ten recorded validation points. No smoothing or added points; lines connect observations within each stage. The Stage B detail panel has a different vertical scale. |
| `posttraining_tradeoff` | [POSTTRAINING_TRADEOFF.csv](../tables/POSTTRAINING_TRADEOFF.csv) | Five snapshots from the same historical diagnostic comparison. P3 step 320 is the blend parent; step 640 is shown for context. No continuous interpolation curve is inferred. |

The first figure shows reference validation loss; minibatch training losses use a
different, changing sample and are not mixed into this curve. The second figure
uses a reused development set with correlated presentations; it does not rank
general assistant quality. The 90% line marks the study's procedural target.

[Source metadata and hashes](../provenance/REPORT_DATA_PROVENANCE.json) identify
the existing records from which these aggregates were extracted. The figures
are redraws of recorded results, not new training or evaluation runs.
