# Frozen analysis plan: independent confirmation of the n=500 LRT null cell

**Frozen before the confirmation run; 2026-09-07.**  This plan was written
after seeing the primary result of 14 rejections among 150 analysis-valid
replicates at the prespecified standardized n=500 null DGP.  It is therefore
an openly outcome-prompted *independent confirmation*, not a second primary
analysis and not a claim that the confirmation question was specified before
the primary run.  All choices below are fixed before any confirmation data are
generated or inspected.

## Question and status of the original result

The primary 1,200-row LRT operating-characteristic experiment is immutable and
remains the primary result.  Its n=500, separation-zero cell gave 14/150 =
9.33% rejections (95% Clopper--Pearson interval 5.20%--15.16%; 95% Wilson
interval 5.64%--15.06%).  This isolated point estimate is higher than the
nominal 5% rate, with substantial Monte Carlo uncertainty.  The confirmation
asks whether an independent, more precise run supports an elevated null-DGP
rejection probability for the *implemented algorithm*.

Neither experiment establishes composite-null supremum size, the null law of
the ideal exact/global LR, or the LR-scale approximation condition in the
conditional bootstrap theorem.  The target is the empirical rejection
probability of the fully specified implemented statistic under one
standardized null DGP.

## Frozen design

- DGP: 600 independent samples of size n=500 from N(0,1), exactly the
  separation-zero generator in the primary experiment.
- Statistic: `bootstrap-calibrated implemented approximate likelihood-ratio
  statistic` under protocol
  `analytic-gaussian-null-explicit-candidate-v1`.
- Restricted fit and score: analytic Gaussian MLE and its closed-form Gaussian
  log likelihood; zero numerical null starts.
- Alternative: fixed m=2 two-center WFCM endpoint, defensive proposal,
  M=max(4000,15n)=7500 importance draws, and a 450-iteration cap.  Here
  `n_init=3` means exactly three numerical optimization starts in total: the
  embedded-null initialization followed by two seeded k-means++ starts.  The
  exact embedded null retained in the final scored candidate set is not a
  fourth optimizer start.
- Final scoring: deterministic adaptive one-dimensional quadrature with base
  grid 100,001.  The scored candidate is the maximum of the numerical
  alternative endpoint and the exact coincident-center embedding of the
  analytic null.
- Calibration: B=199 fitted-null parametric-bootstrap samples per outer
  dataset.  Every bootstrap sample repeats the same alternative fitting,
  deterministic scoring, and candidate-maximum rule.
- P-value: (1 + number of convergence-valid bootstrap statistics greater than
  or equal to the observed statistic)/(1 + B_valid).  Ties at zero are
  included.  Although the shared low-level engine can calculate a p-value
  after 80% of refits, this confirmation's locked publication rule is
  stricter: an outer analysis is confirmation-valid only when the observed
  calculation is valid and at least ceil(0.95*199)=190 bootstrap refits are
  valid.  Only this 95% indicator enters X.
- Rejection indicator: bootstrap p-value <=0.05.  The confirmation count X is
  the sum of these 600 indicators.

The driver imports the primary experiment's LRT engine unchanged.  Its
LRT-specific source fingerprint is pinned.  A separate prelaunch manifest also
pins the driver, engine/core sources, test file, this plan, and the write-back
template.  The extra confirmation fields are passive instrumentation: they do
not draw random numbers, alter arguments or returned fit objects, or affect the
statistic or decision.  Each row records the attempted/returned fit-call counts
needed to audit that the observed call and all B bootstrap calls were seen.
Publication execution hard-selects joblib's `loky` process backend, so the
temporary per-worker instrumentation wrapper cannot be shared across threads.

## Randomness and independence audit

The primary LRT used base seed 20260907.  Its full schedule contains 242,400
direct random-stream instances (data, alternative fit, bootstrap-data parent,
and bootstrap alternative-fit streams); because these were represented by
32-bit child seeds, 242,386 values are globally unique.  The 14 global
birthday collisions occur across different outer analyses, not within one
outer analysis, and are already disclosed in the primary raw audit.

The confirmation uses master seed 20260907001 and a frozen deterministic
collision-avoidance map.  For outer replicate r, beginning at
20260907001+10000r, it selects the smallest root whose 202 child seeds are
unique, disjoint from all earlier confirmation child seeds, and disjoint from
all 242,386 primary child-seed values.  The resulting 121,200 confirmation
child streams are globally unique, have zero overlap with the primary, and
have schedule SHA-256
`0c4866173ebce60c4d4b098da5c62b2eb392eb99ab98b6d186e4d588c63928c4`.
The largest deterministic root-search offset is one.  This seed-map upgrade is
an audit improvement only; it does not change the statistic, DGP, optimizer,
or bootstrap algorithm.

## Frozen interpretation

The following thresholds apply only if all 600 scheduled outer analyses are
valid, because they are thresholds for a fixed Binomial denominator of 600:

- X <= 34: the confirmation does **not support elevated null-DGP rejection**.
  This is not acceptance of exact 5% calibration.
- 35 <= X <= 39: **inconclusive**.
- X >= 40: the confirmation **supports elevated null-DGP rejection**.  Under
  a Binomial(600,0.05) benchmark, P(X >= 40)=0.04205.

If even one outer analysis is invalid, the fixed-R threshold classification
is `not eligible`; the valid/attempted count and failure diagnostics are
reported, and no replacement replicate is added.  The primary 14/150 result
is always displayed separately.  The confirmation will be displayed as
X/600 with its Monte Carlo standard error and both 95% Clopper--Pearson and
Wilson intervals.  A pooled (14+X)/750 estimate may be shown only as a
clearly labeled secondary descriptive summary; it is not used for the frozen
decision.

There are no interim looks, early stopping, data-dependent tuning, selective
reruns of failed replicates, or sequential additions.  No result will be
discarded or replaced because its direction is inconvenient.  The full driver
accepts only the canonical output path and requires that path to be absent--an
existing empty directory is rejected along with a nonempty one.  It has no
resume or overwrite mode, and a smoke run cannot occupy the canonical path.
A genuine machine interruption before creation of any raw-result file may be
archived as an incomplete technical run and restarted from the identical seeds
and settings only after verifying that no raw-result file was created; its
output may not be inspected or combined.  A completed or raw-producing run may
never be moved aside to make a second launch possible.

## Mandatory output and audit fields

Each raw row retains all primary LRT diagnostics and additionally stores the
integer bootstrap exceedance count, B_valid, the fixed `numpy linear`
empirical 0.90/0.95/0.99 bootstrap-statistic quantiles, observed
statistic-minus-raw difference, fitted alternative scale/centers/weights/m,
minimum absolute and relative free-coordinate distance to the optimization
box, projected and raw gradient infinity norms, observed-plus-bootstrap
fit-call attempt/return counts, and bootstrap statistic zero,
endpoint-fallback, and material-fallback counts.  Individual bootstrap
statistics are not saved; these sufficient audit aggregates keep the output
manageable.

Publication eligibility requires the exact row grid and schema, all algebraic
and plus-one p-value identities, monotone quantiles, valid count bounds, the
collision-free/disjoint seed audit, matching source manifest and pinned LRT
fingerprint, all 600 outer analyses valid, and byte-identical primary-result
fingerprints before and after the run.

The summary and machine-readable write-back store, separately for the primary,
confirmation, and descriptive pooled rows, the attempted, analysis-valid,
failed, and rejection counts; the denominator kind; the decimal and formatted
rate; and numeric plus formatted 95% Clopper--Pearson and Wilson endpoints.
The confirmation block also stores its Monte Carlo standard error, fixed-R
eligibility flag, classification, and exact Binomial(600,0.05) upper-tail
benchmark.  Thus no survivor denominator or CI convention is implicit.

## One-time launch command

This command must be launched only after the single-worker coverage experiment
has finished, so the active computation is not contended:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m simulations.lrt_null_confirmation \
  --profile full --jobs 4 \
  --output revision_work/results/inference_lrt_null_confirmation
```
