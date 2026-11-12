# Figures 10–12 and Table 6 with the November Algorithm 1

This completed rerun uses the one-dimensional `additional` setting and the November 8 application's original `fit_fixed`. All 2,000 prescribed dataset tasks are saved: 400 interval experiments, 1,000 estimation experiments, and 600 complete seven-point m grids. All satisfy the original numerical acceptance policy. This is a numerical status, not a guarantee of global likelihood maximization or statistical accuracy.

The three scientific figures and Table 6 are in [`results/summary/algorithm1_1d/`](results/summary/algorithm1_1d/). Raw outcomes, configuration, environment, numerical source snapshots, and execution history are in [`results/algorithm1_1d_full_2026-11-12/`](results/algorithm1_1d_full_2026-11-12/). The earlier `additional/` and coverage archives remain historical results of their original fitter; they are not the sources for these replacement figures and table.

## Model and fitting

The generating density is the normalized one-dimensional WFCM model with centers (-2, 2), weights (0.35, 0.65), sigma=0.9, and m=2. The weights are distance penalties, not mixture probabilities. Independent uniforms are inverted through a numerical CDF constructed on a 200,001-point grid with extended tails.

`simulations/algorithm1_fit.py` calls only `manuscript_engine.fit_fixed` to estimate parameters. The latter's nine fitting/helper functions have identical Python syntax trees to `FCM_breast.ipynb` at commit `5d15846d26cbd9b866658ad8fa153e97cbfbbf54`. Its original block updates and final joint likelihood refinement are retained. There is no direct-likelihood preliminary fit, added safeguard, new parameter bound, replacement seed, or repair fit.

Observed fits use three seeded random-observation center initializations, uniform weights, and initial sigma=1. A bootstrap refit uses one observed-fit warm start. The outer limit is 2,000, epsilon=delta=1e-5, the scale/weight limit is 1,000 per block, and the total joint-refinement budget is 1,000 iterations. The lowest-NLL endpoint is selected, then the original application's checks are applied: block convergence, no failed block optimizations, and final refinement success or its declared iteration limit at an accepted finite endpoint.

Each fit uses the existing iid defensive proposal: 90% fitted two-component GMM plus 10% Student-t5, centered at the sample mean with scale four times the regularized sample covariance. The integration cloud stays fixed within a fit. Candidate m values share a cloud and initialization seeds within each dataset.

## Designs and presentation

| Output | Fixed design | Display |
|---|---|---|
| Figure 10 | 250 datasets at each n=200,500,1000,2000; fixed m=2; M=max(2000,ceil(n^1.5)) | Mean aligned center RMSE, absolute scale error, L1 weight error; ±1.96 dataset MCSE; log-log axes and n^-1/2 reference |
| Figure 11 | Same 1,000 fits; independent truth Fisher information from 750,000 draws | Four marginal QQ panels for log(sigma), v1, v2, log(w1/w2); same-color points connected in quantile order, with all sample sizes overlaid |
| Figure 12 | 200 datasets at each n=200,500,1000; m grid 1.5,1.7,1.9,2.0,2.1,2.3,2.5 | Mean total training NLL with ±1.96 MCSE bars in the upper panels; the same mean curves enlarged without error bars below; one column per n, with the minimum mean NLL marked |
| Table 6 | 200 datasets at each n=200,500; 199 parametric and 199 nonparametric refits each; M=max(4000,20n) | Normal bootstrap/Fisher upper panel, percentile bootstrap/Fisher lower panel; coverage, length and CPU seconds |

Normal-bootstrap intervals are centered at the bootstrap mean with 1.96 times the bootstrap SD, using denominator B. Percentile intervals use empirical 0.025 and 0.975 quantiles. Fisher information uses 50,000 fitted-model draws and the existing eigenvalue rule, omitting finite-IS covariance. All 159,200 prescribed bootstrap refits are valid. Both timing panels now use CPU seconds, including the observed fit, rather than the previous mixture of CPU and wall-clock time.

Figure 12 uses pointwise Monte Carlo standard errors: the sample SD of total training NLL across R=200 independent datasets divided by sqrt(R). Its ±1.96 MCSE bars describe precision of the estimated mean. The lower panels enlarge the identical total-NLL means without error bars. Selection frequencies are saved separately. No candidate is selected from an incomplete or invalid grid.

## Findings and limitations

Mean estimation errors decrease with n. Nevertheless, one n=500 fit and one n=1000 fit have near-zero scale and a weight near the simplex boundary despite passing the original numerical checks. They remain in every mean, error bar and QQ plot. Figure 11 shows the entire range, with central-range insets using the same quantile ranks. The results do not demonstrate uniformly accurate Gaussian approximation across all four sample sizes.

Mean NLL minima occur at m=1.9,2.0,1.9 for n=200,500,1000. Individual selections remain variable: exact m=2 is chosen in 3.0%,9.0%,9.5% of datasets; the two grid endpoints together account for 72.5%,60.5%,33.5%. This experiment does not establish reliable recovery of the exact generating m at these sample sizes.

Normal-bootstrap center coverage is 93.5% and 96.2%, versus Fisher coverage 92.2% and 95.8%. Percentile coverage is 94.0% and 96.0%. Coverage and length average the two aligned centers within each dataset; datasets are the independent MCSE units. Rounding matches the LaTeX table.

The frozen raw estimation records include diagnostic coordinates calculated by the shared parameter packer, which clips weights at 1e-12 even with a zero model weight floor. This affects one saved weight log-ratio. The reporting script reconstructs the published QQ coordinates directly from the actual fitted positive weights, without clipping, refitting, or editing the raw records. The independent audit checks both the original saved diagnostic and the exact reconstructed coordinate.

## Reproduction

Use Python 3.12.3 for the recorded environment. From `revision_simulations`:

```sh
python -m pip install -r requirements.txt -r results/algorithm1_1d_full_2026-11-12/requirements-reproduction.txt
python -m reporting.algorithm1_results --run-dir results/algorithm1_1d_full_2026-11-12 --output runs/algorithm1_summary
```

Reporting performs no fitting and refuses partial full-run task sets. To repeat the numerical experiments in a new output folder:

```sh
python -m simulations.algorithm1_experiments --profile smoke --jobs 8 --output runs/algorithm1_pilot
python -m simulations.algorithm1_experiments --profile full --jobs 8 --pilot runs/algorithm1_pilot --output runs/algorithm1_full
```

The successful pilot must contain exactly all 15 prescribed tasks and the same numerical source hashes. Failed pilot outcomes block full execution. Every completed full-run outcome, including a recorded failure, is reused on resume; no replacement is performed.

If changing parallelism, stop the previous controller and verify its workers have exited before using:

```sh
python -m simulations.algorithm1_schedule --jobs 8 --pilot runs/algorithm1_pilot --output runs/algorithm1_full
```

The archived run began with four workers and continued with eight, preserving 15 completed datasets byte-for-byte. Each worker used one numerical-library thread on an eight-core, 24-GiB machine. `run_config.json` preserves the initial worker specification; `executions/` records actual concurrency and checkpoint hashes. The local working-commit identifier in `environment.json` is provenance; the bundled source snapshots and hashes identify the numerical code independently of the clean publication branch.

## Independent checks

The scripts below call no optimizer. They reconstruct intervals, coverage and MCSEs, recompute truth Fisher information from independently derived m=2 energy derivatives, regenerate saved data/clouds, rescore every selected estimation and grid endpoint, and compare all four figure summary tables.

```sh
PYTHONPATH=. python results/provenance/algorithm1_1d/verify_coverage_independent.py results/algorithm1_1d_full_2026-11-12 runs/coverage_audit
PYTHONPATH=. python results/provenance/algorithm1_1d/verify_additional_independent.py results/algorithm1_1d_full_2026-11-12 results/summary/algorithm1_1d runs/additional_audit
python -m pytest -q
```

Recorded audit outputs are in [`results/provenance/algorithm1_1d/final/`](results/provenance/algorithm1_1d/final/). Real-data notebooks/results and unrelated simulation studies are unchanged.
