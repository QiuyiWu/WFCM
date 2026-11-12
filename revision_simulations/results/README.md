# Simulation results

This folder contains simulation records, summary tables, figures, and verification files for the paper.

## Folder contents

- [algorithm1_1d_full_2026-11-12/](algorithm1_1d_full_2026-11-12/): individual simulation results, experiment settings, and software versions for Figures 10–12 and Table 6.
- [algorithm1_1d_pilot_v2_2026-11-12/](algorithm1_1d_pilot_v2_2026-11-12/): a small test run of the simulation procedure.
- [summary/algorithm1_1d/](summary/algorithm1_1d/): Figures 10–12, Table 6, and their numerical summaries.
- [provenance/algorithm1_1d/final/](provenance/algorithm1_1d/final/): checks of the saved estimates, confidence intervals, and summary calculations.
- [provenance/primary_lrt/](provenance/primary_lrt/): the source code used for the likelihood-ratio experiments.

## Summary files

The following files are in `summary/`:

| Files | Contents |
|---|---|
| `benchmark.csv`, `benchmark.png` | Comparison of nine methods under two simulation settings |
| `selection.csv` | Cluster-number selection and confidence-interval coverage |
| `dimension_scaling.csv`, `sample_size_scaling.csv`, `mc_size_sensitivity.csv`, `computation.png` | Accuracy and computing time across dimensions, sample sizes, and Monte Carlo sample sizes |
| `coverage.csv` | Confidence-interval coverage, length, and computing time |
| `lrt.csv` | Null rejection rates and power, with the independent confirmation reported separately |
| `additional/` | Consistency, normality, and fuzziness-selection summaries, with a combined figure |

Simulation records are stored as compressed JSON or CSV files. In CSV summaries, proportions range from 0 to 1, and times are in seconds.

See the [reproduction guide](../ALGORITHM1_1D_RERUN.md) for instructions on running the simulations and generating the figures and tables.
