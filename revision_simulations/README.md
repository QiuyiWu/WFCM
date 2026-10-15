# Revision simulations

Code and saved results for the revision of *Statistical Inference for Fuzzy Clustering*. This folder contains the simulation drivers, original run configurations, dataset-level results, and scripts for rebuilding the summary tables and figures. The original notebooks remain in the repository root.

## Experiments

| Experiment | Paper section | Design and purpose |
|---|---|---|
| `benchmark` | Appendix D.1: method comparison | Nine clustering methods; 100 datasets per geometry–imbalance–overlap cell, with n = 600 and d = 2 |
| `selection` | Appendix D.2: selecting k | 250 datasets each at n = 150, 300; compares k = 2, 3, 4 and evaluates coverage after selection |
| `computation` | Appendix D.3: importance sampling and timing | 20 datasets per configuration; varies dimension, proposal distribution, Monte Carlo budget, and sample size |
| `coverage` | Appendix D.4: bootstrap and Fisher intervals | 200 datasets each at n = 200, 500; compares nominal 95% center intervals using 199 bootstrap samples |
| `lrt` | Appendix D.5: null rejection and power | 150 datasets per sample-size–separation cell; evaluates bootstrap-calibrated rejection rates |
| `confirmation` | Appendix D.5: independent null check | 600 additional null datasets at n = 500, using the same testing procedure and 199 bootstrap samples |
| `additional` | Supplementary experiments | Consistency, normality, and m-grid sensitivity in a one-dimensional WFCM model |

The benchmark compares WFCM, standard FCM, polynomial FCM, k-means, equilibrium k-means, HDBSCAN, Ward clustering, spectral clustering, and a full-covariance Gaussian mixture. Paper-facing summaries use spherical and anisotropic data, cluster-size ratios of 1:1, 4:1, and 10:1, and three overlap levels.

The selection experiment uses the weighted Xie–Beni criterion. It records how often each k is selected, coverage with k fixed at its true value, and coverage following selection.

Computational experiments consider dimensions 2, 10, 25, and 50; full-covariance, diagonal-covariance, and defensive proposals; and importance-sampling budgets M/n = 1, 5, 20, and 80. Sample-size scaling is evaluated at n = 250, 1,000, and 4,000.

For the LRT experiments, Δ = 0 denotes data generated from the standard Gaussian null. Positive Δ denotes the distance between the two generating centers, so the corresponding rejection rate estimates power. The independent null confirmation is reported separately from the primary experiment.

The additional experiments use 250 datasets per sample size for estimation and normality checks, and 200 datasets per sample size for m-grid selection. Their outputs are grouped under `additional/`.

## Folder structure

```text
revision_simulations/
├── README.md
├── requirements.txt
├── run.py                   # Common entry point for simulation runs
├── simulations/             # Numerical implementation and experiment drivers
│   └── tests/               # Numerical and release checks
├── reporting/
│   └── summarize.py         # Rebuild tables and figures from saved records
└── results/
    ├── */                   # Raw results and original run configurations
    ├── provenance/          # Preserved source and analysis records
    └── summary/             # Ready-to-view tables and figures
```

New runs and regenerated summaries are written to `runs/`, which is excluded from version control.

## Quick start

From the repository root, using Python 3.12:

```bash
cd revision_simulations
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reporting/summarize.py
```

The last command reads the saved results, verifies their checksums, and rebuilds 11 CSV tables and three figures in `runs/summary/`. It does not rerun the simulations. Ready-to-view copies are available in [`results/summary/`](results/summary/).

To choose another summary location:

```bash
python reporting/summarize.py --output runs/rebuilt_summary
```

## Running simulations

For a short end-to-end check:

```bash
python run.py coverage
```

`run.py` accepts any experiment listed above. The default `smoke` profile uses a small simulation budget to check the workflow. The `full` profile uses the original experiment settings and seeds:

```bash
python run.py benchmark --profile full
python run.py selection --profile full
python run.py computation --profile full
python run.py coverage --profile full
python run.py lrt --profile full --jobs 4
python run.py confirmation --profile full --jobs 4
python run.py additional --profile full
```

Each command writes its raw records and experiment summaries to `runs/<experiment>_<profile>/`. To repeat a run, choose a new output directory:

```bash
python run.py coverage --output runs/coverage_check_2
```

Timed experiments—`benchmark`, `computation`, `coverage`, and `additional`—use one worker. LRT and confirmation runs support multiple workers through `--jobs`. Full bootstrap experiments can take several hours.

The reporting script rebuilds summaries from the bundled `results/` archive; newly simulated results remain in their own run directories.

## Reading the outputs

The main files in [`results/summary/`](results/summary/) are:

- `benchmark.csv`: adjusted Rand index (ARI), label-matched accuracy, minority-cluster recall, prototype RMSE, runtime, and valid-fit counts.
- `selection.csv`: selection counts, fixed-k coverage, coverage conditional on selecting the true k, and the probability of both selecting the true k and covering a center.
- `dimension_scaling.csv`, `sample_size_scaling.csv`, and `mc_size_sensitivity.csv`: effective sample size, maximum importance weight, center error, likelihood differences, and runtime.
- `coverage.csv`: interval coverage, average length, and computation time for bootstrap and model-Fisher intervals.
- `lrt.csv`: rejection counts, rejection rates, and Monte Carlo standard errors for the primary experiment and independent confirmation.
- `additional/`: estimation errors, normality summaries, m-grid likelihood comparisons, and selection frequencies.

The three figures summarize method-comparison ARI, computational scaling, and the additional estimation and m-grid experiments.

Proportions in the CSV files are on the 0–1 scale; runtimes are in seconds. Columns ending in `_mcse` contain Monte Carlo standard errors, and `_defined` columns give the number of observations contributing to a summary. Computation tables report median runtime and its interquartile range.

## Implementation and reproducibility

`simulations/wfcm_core.py` fits the normalized WFCM likelihood using multi-start L-BFGS-B and a fixed importance-sampling cloud within each fit. The experiment drivers retain the settings used for the saved runs. In the method benchmark, polynomial FCM uses beta = 0.5, and equilibrium k-means is implemented through `sklekmeans`.

For interval comparisons, coverage and length are averaged over the two aligned centers within each dataset before computing Monte Carlo standard errors. Bootstrap refits rebuild the proposal and importance-sampling cloud. Reported interval times include the observed-data fit and the corresponding interval calculation.

Monte Carlo budget comparisons score fitted endpoints on a common independent cloud and use M/n = 80 as the reference. Benchmark prototype RMSE measures Euclidean center error; computational experiments report center-coordinate RMSE.

Each result folder contains its original `run_config.json`. Raw CSVs are compressed without changing their contents, and the original LRT source snapshot and confirmation records are retained under [`results/`](results/README.md). Timing summaries use the saved execution times.

To run the numerical and release checks:

```bash
python -m pytest -q
```

The tests cover the numerical implementation, experiment drivers, output handling, and agreement between the saved records and reported summaries.
