# Primary full-LRT launch-source snapshot

Date preserved: 2026-09-07

## Purpose

This directory preserves the exact Python sources recorded by
`revision_work/results/inference_lrt_full/source_manifest.sha256` when the
primary full LRT run was launched.  It is archival evidence for that result;
it is not the current selection driver.

## Reconstruction

The launch-time `inference_experiment.py` was reconstructed from the current
post-selection driver, whose SHA-256 is
`ae7ce6cca4a7f3c51c1da2b4810db3484c7b494d6666e981d8fd14e59d3866a9`,
by reversing only the two post-LRT selection patches documented in
`revision_work/selection_design_audit.md`:

1. introduction of named selection-design and seed-path constants in place
   of the equivalent literals;
2. exact reuse of `fixed_analysis` when WXBI selects `k=2`, instead of a
   second Fisher calculation with seed tag 25; and
3. the corresponding selection protocol fields added to `run_config.json`.

No LRT, coverage, likelihood, bootstrap, optimizer, scoring, or profile code
was altered in the reconstruction.  The complete forward difference from the
archived launch driver to the current driver is preserved as
`post_lrt_selection_only.diff`; inspection shows that every hunk is confined
to selection constants, `_selection_one`, `summarize_selection`, or selection
metadata in `main`.

The reconstructed file was accepted only after its byte-level SHA-256 exactly
matched the launch manifest:

```text
e4014443221b9ae27d74ed1cf5634111ba8c9ed21822c6de1ab2ad54ee7c385c  inference_experiment.py
```

The matching `wfcm_core.py` and `__init__.py` were copied from the current
tree only after their bytes were independently checked against the same
launch manifest:

```text
3815ea87e7dc6564a0d349165b2ce207b2a818a54025a9b14afeffd82e6412a8  wfcm_core.py
c003e8537e67706ae3cd37f89fba9ffb383362ccf6236a5864e1150787f1eb62  __init__.py
```

The archived three-file manifest in this directory deliberately retains the
original project-relative names used by the full-run manifest.  The manifest
in `revision_work/results/inference_lrt_full/` has the same three hash/name
pairs.

## Verification recipe

From the project root:

```text
shasum -a 256 \
  revision_work/source_snapshots/primary_lrt_e4014443/inference_experiment.py \
  revision_work/source_snapshots/primary_lrt_e4014443/wfcm_core.py \
  revision_work/source_snapshots/primary_lrt_e4014443/__init__.py
```

The expected hashes are the three values above.  Applying
`post_lrt_selection_only.diff` to the archived driver produces the current
`ae7ce6cc...` driver; reversing that diff reconstructs this archived file.
