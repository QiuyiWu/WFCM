# Revision simulations

Code and saved results for the revision of *Statistical Inference for Fuzzy Clustering*.
The original notebooks remain in the repository root.

## Experiments

| Experiment | Paper section | Saved design |
|---|---|---|
| `benchmark` | Appendix D.1: method comparison | Nine methods; 100 datasets per geometry–imbalance–overlap cell |
| `selection` | Appendix D.2: selecting k | 250 datasets each at n = 150, 300 |
| `computation` | Appendix D.3: importance sampling and timing | 20 datasets per configuration |
| `coverage` | Appendix D.4: bootstrap and Fisher intervals | 200 datasets each at n = 200, 500; 199 bootstrap samples |
| `lrt` | Appendix D.5: null rejection and power | 150 datasets per sample-size–separation cell |
| `confirmation` | Appendix D.5: independent null check | 600 datasets at n = 500; 199 bootstrap samples |
| `additional` | Supplementary experiments | Consistency, normality and m-grid sensitivity in a one-dimensional WFCM model |

The paper-facing benchmark uses spherical and anisotropic data. The full saved
benchmark also retains the two-moons diagnostic. The additional experiments
have their own summaries; the original notebook figures remain with `FCM_weight.ipynb`.

## Quick start

From the repository root, using Python 3.12:

```bash
cd revision_simulations
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reporting/summarize.py
```

This rebuilds the tables and three compact figures in `runs/summary/` from the
saved records, without rerunning simulations. The checked-in versions are in
[`results/summary/`](results/summary/).

For a short end-to-end check, run:

```bash
python run.py coverage
python -m pytest -q
```

`run.py` accepts any experiment in the table. It defaults to a small `smoke`
profile; `--profile full` uses the original design and seeds:

```bash
python run.py benchmark --profile full
python run.py lrt --profile full --jobs 4
```

New runs go to `runs/<experiment>_<profile>/`. Use `--output PATH` for another
location. Timed comparisons use one worker; full bootstrap experiments can take
several hours.

## Implementation and results

`simulations/wfcm_core.py` fits the normalized WFCM likelihood by multi-start
L-BFGS-B with a fixed importance-sampling cloud. The drivers retain the settings
used for the saved runs. The benchmark includes polynomial FCM with beta = 0.5
and equilibrium k-means through `sklekmeans`.

The reporting script computes summaries directly from compressed raw CSVs.
Coverage and interval length are averaged over the two aligned centers within
each dataset before computing Monte Carlo standard errors. LRT tables use the
bootstrap-calibrated statistic and report the primary run and independent
confirmation separately. Timing values are from the saved runs.

Each result folder contains its run configuration. Checksums and the original
LRT source snapshot are included under [`results/`](results/README.md).
