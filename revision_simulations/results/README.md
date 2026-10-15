# Saved results

Each experiment folder contains the original raw CSV, compressed without
changing its contents, and the recorded run configuration. `manifest.sha256`
checks these files before summaries are generated. Repeated JSONL exports and
intermediate checkpoints are omitted.

`summary/` contains tables rebuilt from those records and three figures:

- `benchmark.csv` and `benchmark.png`: the two paper geometries, with means,
  MCSEs and valid-fit counts for all nine methods.
- `selection.csv`: selection counts, fixed-k coverage, conditional coverage,
  and the probability of selecting the true k and covering a center.
- `dimension_scaling.csv`, `sample_size_scaling.csv`, `mc_size_sensitivity.csv`
  and `computation.png`: importance weights, endpoint error and timing.
- `coverage.csv`: dataset-level coverage, interval length and total time.
- `lrt.csv`: rejection counts and binomial MCSEs, with confirmation separate.
- `additional/`: consistency, normality and m-grid summaries and a combined figure.

Proportions in CSVs are on the 0–1 scale and times are in seconds. Benchmark
prototype RMSE uses Euclidean center error; computation summaries use
center-coordinate RMSE. Monte Carlo budget comparisons use the same dataset
and independent scoring cloud, with M/n = 80 as the reference endpoint.

The original runs used Python 3.12.3 on macOS ARM64. Recorded package versions
are retained in the configurations that captured them. The numerical source
files are unchanged; the package initializer omits an unused generalized-MM
import, and `run.py` supplies the release paths and original seeds.

`provenance/primary_lrt/` preserves the launch-time inference source. Subsequent
changes affected only the k-selection calculation; the LRT engine fingerprint
is checked by the release tests. The independent confirmation's analysis plan
and completed audit are retained alongside its records. The release runner
reuses its numerical engine and seed schedule, writing a new run directory.
Paths in the preserved provenance notes refer to the original working project.
