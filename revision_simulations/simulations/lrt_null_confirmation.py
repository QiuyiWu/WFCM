"""Independent, fixed-design confirmation of the n=500 LRT null cell.

This module is intentionally separate from :mod:`simulations.inference_experiment`.
It calls that module's locked ``_lrt_one`` implementation without changing the
statistic, optimizer, bootstrap, or primary 1,200-row result.  The wrapper adds
only audit fields that have no effect on fitting or rejection decisions.

The publication confirmation is fixed at n=500, R=600, B=199, three
alternative starts, and the v1 analytic-null/explicit-candidate protocol.  It
is a one-time diagnostic: the primary run remains primary, the two runs are
not pooled for the prespecified interpretation, and no sequential extension
is permitted after the confirmation count is observed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta, binom, norm

import simulations.inference_experiment as inference


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "revision_work" / "results" / "inference_lrt_null_confirmation"
)
PRIMARY_LRT_OUTPUT = ROOT / "revision_work" / "results" / "inference_lrt_full"

CONFIRMATION_PROTOCOL_VERSION = "lrt-null-n500-r600-fixed-once-v1"
CONFIRMATION_N = 500
CONFIRMATION_SEPARATION = 0.0
CONFIRMATION_REPS = 600
CONFIRMATION_BOOTSTRAP_REPS = 199
CONFIRMATION_ALTERNATIVE_N_INIT = 3
CONFIRMATION_MAXITER = 450
CONFIRMATION_IMPORTANCE_SIZE = 7_500
CONFIRMATION_MIN_BOOTSTRAP_VALID_FRACTION = 0.95
CONFIRMATION_MIN_BOOTSTRAP_VALID = 190

# This seed is deliberately outside the 32-bit range used by the primary
# run's stored child seeds.  It is the start of a deterministic search for
# per-replicate roots whose *every direct child stream* is also disjoint from
# every direct child stream in the complete primary LRT design.
CONFIRMATION_MASTER_SEED = 20_260_907_001
CONFIRMATION_ROOT_STRIDE = 10_000
PRIMARY_LRT_BASE_SEED = 20_260_907
PRIMARY_LRT_NS = (200, 500)
PRIMARY_LRT_SEPARATIONS = (0.0, 1.0, 2.0, 3.0)
PRIMARY_LRT_REPS = 150
PRIMARY_LRT_BOOTSTRAP_REPS = 199
PRIMARY_PINNED_ARTIFACT_SHA256 = {
    "lrt_raw.csv": "619b65c3563ce29a21bcbe45f24d2488e374b97c1eb3134554b3e5c8979302c4",
    "lrt_raw.jsonl": "e8d814d657a1b4a78f832cea9d8001522a75fa11dee0406841d7d29e08df52b5",
    "lrt_summary.csv": "43704831a850de35913af0d0dc544f36e8e176b0c6d83bb9a90b4c783a1cbba9",
    "run_config.json": "e3aa17f463b93ceddc57cb339ca63e424d1057d6064eef6cf7f84b5c0fc4c9c4",
    "source_manifest.sha256": "305d45f02a0bde77182bd48fdcea45b45cccb2add61557a68aebd0dc001c935d",
    "source_provenance.md": "3afc6648d23df07947280950540cb759bf1bf823e3317b1aa46d8c6b8f32a04b",
}

FROZEN_SOURCE_MANIFEST = (
    ROOT / "revision_work" / "lrt_null_confirmation_frozen_manifest.sha256"
)
FROZEN_SOURCE_NAMES = (
    "simulations/lrt_null_confirmation.py",
    "simulations/inference_experiment.py",
    "simulations/wfcm_core.py",
    "simulations/__init__.py",
    "simulations/tests/test_lrt_null_confirmation.py",
    "revision_work/lrt_null_confirmation_preregistration.md",
    "revision_work/lrt_null_confirmation_writeback_template.tex",
)

SUPPORT_THRESHOLD_MAX = 34
INCONCLUSIVE_THRESHOLD_MIN = 35
INCONCLUSIVE_THRESHOLD_MAX = 39
ELEVATION_THRESHOLD_MIN = 40

RAW_STEM = "lrt_null_confirmation"
PRIMARY_ARTIFACT_NAMES = (
    "lrt_raw.csv",
    "lrt_raw.jsonl",
    "lrt_summary.csv",
)
CONFIRMATION_ARTIFACT_NAMES = (
    f"{RAW_STEM}_raw.csv",
    f"{RAW_STEM}_raw.jsonl",
    f"{RAW_STEM}_summary.csv",
    "confirmation_audit.json",
    "run_config.json",
    "seed_schedule_audit.json",
    "source_manifest.sha256",
    "source_provenance.md",
    "writeback_fields.json",
)

ENGINE_OBJECT_NAMES = (
    "_spawn_seed",
    "importance_size",
    "_log_normalizer_grid_1d",
    "exact_loglik_1d",
    "normal_null_fit",
    "_null_embedded_start",
    "fit_alternative_for_lrt",
    "_lrt_one",
)

# Filled from the canonical serialization returned by
# ``lrt_engine_fingerprint``.  A future edit to any LRT-defining object makes
# a publication-profile launch fail until it receives a new explicit audit.
LOCKED_LRT_ENGINE_SHA256 = (
    "60f4c6047ca5cdf1dc2ad2d0f3974ef705195ed5a1f097bddd2910cfeca39953"
)

REQUIRED_BASE_ROW_COLUMNS = (
    "experiment",
    "statistic_name",
    "lrt_protocol_version",
    "n",
    "separation",
    "replicate",
    "data_seed",
    "fit_seed",
    "importance_size",
    "scoring_method",
    "scoring_grid_size",
    "scoring_base_grid_size",
    "scoring_adaptive_local_refinement",
    "scoring_tail_sd_multiplier",
    "null_estimator",
    "null_scoring_method",
    "null_optimizer_used",
    "alternative_candidate_rule",
    "analytic_null_loglik",
    "fitted_alternative_loglik",
    "analytic_null_mean",
    "analytic_null_normal_sd",
    "analytic_null_wfcm_sigma",
    "scored_candidate_selected",
    "statistic",
    "raw_statistic",
    "embedded_null_fallback_used",
    "materially_negative_raw",
    "negative_raw_magnitude",
    "negative_raw_tolerance",
    "fitting_objective_raw_statistic",
    "fitting_objective_nested_within_tolerance",
    "fitting_objective_materially_negative",
    "fitting_objective_negative_tolerance",
    "cross_objective_fallback_used",
    "material_cross_objective_mismatch",
    "bootstrap_pvalue",
    "chi_square_pvalue_diagnostic",
    "bootstrap_reject_05",
    "chi_square_reject_05_diagnostic",
    "fit_success",
    "null_fit_success",
    "analysis_success",
    "gradient_inf_norm",
    "null_gradient_inf_norm",
    "ess_fraction",
    "observed_runtime_seconds",
    "bootstrap_runtime_seconds",
    "bootstrap_requested",
    "bootstrap_finite",
    "bootstrap_valid",
    "bootstrap_optimizer_success",
    "bootstrap_negative_raw_fraction",
    "bootstrap_materially_negative_raw_fraction",
    "bootstrap_minimum_raw_statistic",
    "bootstrap_fitting_objective_negative_fraction",
    "bootstrap_fitting_objective_materially_negative_fraction",
    "bootstrap_minimum_fitting_objective_raw_statistic",
    "bootstrap_cross_objective_fallback_fraction",
    "bootstrap_cross_objective_fallback_count",
    "bootstrap_material_cross_objective_mismatch_fraction",
    "bootstrap_material_cross_objective_mismatch_count",
    "bootstrap_cross_objective_diagnostic_defined",
    "alternative_n_init",
    "null_n_init",
)

REQUIRED_CONFIRMATION_ROW_COLUMNS = (
    "confirmation_protocol_version",
    "confirmation_master_seed",
    "confirmation_replicate_root_seed",
    "confirmation_bootstrap_parent_seed",
    "confirmation_analysis_valid_95",
    "instrumented_fit_call_attempt_count",
    "instrumented_fit_return_count",
    "instrumented_bootstrap_fit_return_count",
    "bootstrap_exceedance_count",
    "bootstrap_empirical_quantile_90",
    "bootstrap_empirical_quantile_95",
    "bootstrap_empirical_quantile_99",
    "bootstrap_quantile_method",
    "bootstrap_statistic_zero_count",
    "bootstrap_statistic_zero_fraction",
    "bootstrap_endpoint_fallback_count",
    "bootstrap_material_endpoint_fallback_count",
    "observed_statistic_minus_raw_statistic",
    "fitted_alternative_sigma",
    "fitted_alternative_center_1",
    "fitted_alternative_center_2",
    "fitted_alternative_weight_1",
    "fitted_alternative_weight_2",
    "fitted_alternative_m",
    "fitted_alternative_raw_gradient_inf_norm",
    "fitted_alternative_min_free_distance_to_bound",
    "fitted_alternative_min_relative_free_distance_to_bound",
)


def publication_profile():
    """Return the exact immutable profile used for the confirmation."""

    base = inference.PROFILES["full"]
    return replace(
        base,
        lrt_reps=CONFIRMATION_REPS,
        lrt_bootstrap_reps=CONFIRMATION_BOOTSTRAP_REPS,
        lrt_ns=(CONFIRMATION_N,),
    )


def smoke_profile():
    """A nonpublication profile for end-to-end software checks only."""

    base = inference.PROFILES["smoke"]
    return replace(base, lrt_reps=1, lrt_ns=(CONFIRMATION_N,))


def minimum_confirmation_bootstrap_valid(profile) -> int:
    return int(
        np.ceil(
            CONFIRMATION_MIN_BOOTSTRAP_VALID_FRACTION
            * int(profile.lrt_bootstrap_reps)
        )
    )


def _canonical_engine_source() -> str:
    pieces = []
    for name in ENGINE_OBJECT_NAMES:
        pieces.append(f"### {name}\n{inspect.getsource(getattr(inference, name))}")
    contract = {
        "alternative_candidate_rule": inference.LRT_ALTERNATIVE_CANDIDATE_RULE,
        "null_estimator": inference.LRT_NULL_ESTIMATOR,
        "null_scoring_method": inference.LRT_NULL_SCORING_METHOD,
        "protocol_version": inference.LRT_PROTOCOL_VERSION,
        "statistic_name": inference.PRIMARY_LRT_NAME,
    }
    pieces.append("### protocol constants\n" + json.dumps(contract, sort_keys=True))
    return "\n".join(pieces)


def lrt_engine_fingerprint() -> str:
    """Hash only the code and constants that define the LRT computation."""

    return hashlib.sha256(_canonical_engine_source().encode("utf-8")).hexdigest()


def validate_locked_engine(*, require_pinned_hash: bool = True) -> dict[str, object]:
    profile = publication_profile()
    checks = {
        "protocol_version": inference.LRT_PROTOCOL_VERSION
        == "analytic-gaussian-null-explicit-candidate-v1",
        "null_estimator": inference.LRT_NULL_ESTIMATOR == "analytic_gaussian_mle",
        "null_scoring": inference.LRT_NULL_SCORING_METHOD
        == "closed_form_gaussian_loglik",
        "candidate_rule": inference.LRT_ALTERNATIVE_CANDIDATE_RULE
        == "max_fitted_alternative_and_embedded_analytic_null",
        "bootstrap_reps": profile.lrt_bootstrap_reps
        == CONFIRMATION_BOOTSTRAP_REPS,
        "alternative_starts": profile.fit_n_init
        == CONFIRMATION_ALTERNATIVE_N_INIT,
        "maxiter": profile.maxiter == CONFIRMATION_MAXITER,
        "importance_size": inference.importance_size(
            CONFIRMATION_N, multiplier=15.0
        )
        == CONFIRMATION_IMPORTANCE_SIZE,
        "sample_size": profile.lrt_ns == (CONFIRMATION_N,),
        "outer_reps": profile.lrt_reps == CONFIRMATION_REPS,
        "null_DGP": CONFIRMATION_SEPARATION == 0.0,
        "strict_bootstrap_validity": (
            CONFIRMATION_MIN_BOOTSTRAP_VALID_FRACTION == 0.95
            and minimum_confirmation_bootstrap_valid(profile)
            == CONFIRMATION_MIN_BOOTSTRAP_VALID
            == 190
        ),
        "fixed_interpretation_thresholds": (
            SUPPORT_THRESHOLD_MAX == 34
            and INCONCLUSIVE_THRESHOLD_MIN == 35
            and INCONCLUSIVE_THRESHOLD_MAX == 39
            and ELEVATION_THRESHOLD_MIN == 40
        ),
    }
    fingerprint = lrt_engine_fingerprint()
    checks["engine_fingerprint"] = (
        fingerprint == LOCKED_LRT_ENGINE_SHA256
        if require_pinned_hash
        else True
    )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError("locked confirmation protocol failed: " + ", ".join(failed))
    return {"checks": checks, "lrt_engine_sha256": fingerprint}


def _direct_lrt_streams(
    base_seed: int,
    *,
    n: int,
    separation: float,
    replicate: int,
    bootstrap_reps: int,
) -> list[tuple[str, int, int]]:
    """Return (role, bootstrap-index, seed) for every direct RNG stream."""

    separation_tag = int(100 * separation)
    streams = [
        ("data", -1, inference._spawn_seed(base_seed, 11, n, separation_tag, replicate)),
        ("alternative_fit", -1, inference._spawn_seed(base_seed, 12, n, separation_tag, replicate)),
        ("bootstrap_data_parent", -1, inference._spawn_seed(base_seed, 13, n, separation_tag, replicate)),
    ]
    streams.extend(
        (
            "bootstrap_alternative_fit",
            b,
            inference._spawn_seed(
                base_seed, 14, n, separation_tag, replicate, b
            ),
        )
        for b in range(bootstrap_reps)
    )
    return streams


@lru_cache(maxsize=1)
def primary_direct_stream_seed_set() -> frozenset[int]:
    """All direct seeds scheduled in the immutable primary 1,200-row LRT."""

    seeds: list[int] = []
    for n in PRIMARY_LRT_NS:
        for separation in PRIMARY_LRT_SEPARATIONS:
            for replicate in range(PRIMARY_LRT_REPS):
                seeds.extend(
                    seed
                    for _, _, seed in _direct_lrt_streams(
                        PRIMARY_LRT_BASE_SEED,
                        n=n,
                        separation=separation,
                        replicate=replicate,
                        bootstrap_reps=PRIMARY_LRT_BOOTSTRAP_REPS,
                    )
                )
    return frozenset(seeds)


@lru_cache(maxsize=None)
def confirmation_root_schedule(reps: int = CONFIRMATION_REPS) -> tuple[int, ...]:
    """Build a deterministic collision-free schedule against the primary run.

    For replicate r, the selected root is the smallest integer beginning at
    ``master + stride*r`` whose 202 direct child streams neither collide with
    the primary schedule nor with an earlier confirmation stream.  The search
    rule, start, and stride are fixed before observing any confirmation data.
    """

    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("reps must be a positive integer")
    used = set(primary_direct_stream_seed_set())
    used.add(PRIMARY_LRT_BASE_SEED)
    roots: list[int] = []
    for replicate in range(reps):
        candidate = CONFIRMATION_MASTER_SEED + CONFIRMATION_ROOT_STRIDE * replicate
        while True:
            streams = _direct_lrt_streams(
                candidate,
                n=CONFIRMATION_N,
                separation=CONFIRMATION_SEPARATION,
                replicate=replicate,
                bootstrap_reps=CONFIRMATION_BOOTSTRAP_REPS,
            )
            child_seeds = [seed for _, _, seed in streams]
            child_set = set(child_seeds)
            if (
                candidate not in used
                and len(child_set) == len(child_seeds)
                and child_set.isdisjoint(used)
            ):
                roots.append(candidate)
                used.add(candidate)
                used.update(child_set)
                break
            candidate += 1
            if candidate >= CONFIRMATION_MASTER_SEED + CONFIRMATION_ROOT_STRIDE * (
                replicate + 1
            ):
                raise RuntimeError("confirmation seed root search exhausted its stride")
    return tuple(roots)


def confirmation_seed_audit(reps: int = CONFIRMATION_REPS) -> dict[str, object]:
    primary = primary_direct_stream_seed_set()
    roots = confirmation_root_schedule(reps)
    confirmation_records: list[tuple[int, int, str, int, int]] = []
    for replicate, root in enumerate(roots):
        confirmation_records.extend(
            (replicate, root, role, bootstrap_index, seed)
            for role, bootstrap_index, seed in _direct_lrt_streams(
                root,
                n=CONFIRMATION_N,
                separation=CONFIRMATION_SEPARATION,
                replicate=replicate,
                bootstrap_reps=CONFIRMATION_BOOTSTRAP_REPS,
            )
        )
    confirmation = [record[-1] for record in confirmation_records]
    canonical = "\n".join(
        ",".join(map(str, record)) for record in confirmation_records
    )
    overlap = set(confirmation).intersection(primary)
    return {
        "primary_base_seed": PRIMARY_LRT_BASE_SEED,
        "primary_stream_instances": (
            len(PRIMARY_LRT_NS)
            * len(PRIMARY_LRT_SEPARATIONS)
            * PRIMARY_LRT_REPS
            * (3 + PRIMARY_LRT_BOOTSTRAP_REPS)
        ),
        "primary_unique_direct_stream_seeds": len(primary),
        "confirmation_master_seed": CONFIRMATION_MASTER_SEED,
        "confirmation_root_count": len(roots),
        "confirmation_unique_root_count": len(set(roots)),
        "confirmation_direct_stream_instances": len(confirmation),
        "confirmation_unique_direct_stream_seeds": len(set(confirmation)),
        "confirmation_primary_direct_seed_overlap_count": len(overlap),
        "confirmation_roots_disjoint_from_primary_direct_seeds": set(roots).isdisjoint(
            primary
        ),
        "confirmation_direct_streams_unique": len(set(confirmation))
        == len(confirmation),
        "confirmation_direct_streams_disjoint_from_primary": not overlap,
        "confirmation_seed_schedule_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "maximum_root_search_offset": max(
            root - (CONFIRMATION_MASTER_SEED + CONFIRMATION_ROOT_STRIDE * replicate)
            for replicate, root in enumerate(roots)
        ),
    }


def classify_confirmation_count(rejection_count: int) -> str:
    """Apply the fixed, nonsequential interpretation thresholds."""

    if isinstance(rejection_count, bool) or not isinstance(rejection_count, int):
        raise TypeError("rejection_count must be an integer")
    if not 0 <= rejection_count <= CONFIRMATION_REPS:
        raise ValueError("rejection_count must be between 0 and 600")
    if rejection_count <= SUPPORT_THRESHOLD_MAX:
        return "does_not_support_elevated_null_rejection"
    if rejection_count <= INCONCLUSIVE_THRESHOLD_MAX:
        return "inconclusive"
    return "supports_elevated_null_rejection"


def _fit_detail(fit: object, values: np.ndarray) -> dict[str, float]:
    detail = {
        "fitted_alternative_sigma": np.nan,
        "fitted_alternative_center_1": np.nan,
        "fitted_alternative_center_2": np.nan,
        "fitted_alternative_weight_1": np.nan,
        "fitted_alternative_weight_2": np.nan,
        "fitted_alternative_m": np.nan,
        "fitted_alternative_raw_gradient_inf_norm": np.nan,
        "fitted_alternative_min_free_distance_to_bound": np.nan,
        "fitted_alternative_min_relative_free_distance_to_bound": np.nan,
    }
    params = getattr(fit, "params", None)
    if params is not None:
        centers = np.asarray(params.centers, dtype=float).reshape(-1)
        weights = np.asarray(params.weights, dtype=float).reshape(-1)
        detail.update(
            {
                "fitted_alternative_sigma": float(params.sigma),
                "fitted_alternative_center_1": (
                    float(centers[0]) if centers.size >= 1 else np.nan
                ),
                "fitted_alternative_center_2": (
                    float(centers[1]) if centers.size >= 2 else np.nan
                ),
                "fitted_alternative_weight_1": (
                    float(weights[0]) if weights.size >= 1 else np.nan
                ),
                "fitted_alternative_weight_2": (
                    float(weights[1]) if weights.size >= 2 else np.nan
                ),
                "fitted_alternative_m": float(params.m),
            }
        )
    diagnostics = getattr(fit, "diagnostics", {})
    raw_gradient = diagnostics.get("raw_gradient_inf_norm", np.nan)
    if np.isfinite(raw_gradient):
        detail["fitted_alternative_raw_gradient_inf_norm"] = float(raw_gradient)
    spec = getattr(fit, "parameterization", None)
    free = getattr(fit, "free_params", None)
    if spec is not None and free is not None:
        free_array = np.asarray(free, dtype=float).reshape(-1)
        bounds = np.asarray(spec.bounds(values), dtype=float)
        if bounds.shape == (free_array.size, 2) and np.all(np.isfinite(bounds)):
            lower_distance = free_array - bounds[:, 0]
            upper_distance = bounds[:, 1] - free_array
            widths = bounds[:, 1] - bounds[:, 0]
            distance = np.minimum(lower_distance, upper_distance)
            relative = distance / widths
            if np.all(np.isfinite(distance)) and np.all(widths > 0.0):
                detail["fitted_alternative_min_free_distance_to_bound"] = float(
                    np.min(distance)
                )
                detail[
                    "fitted_alternative_min_relative_free_distance_to_bound"
                ] = float(np.min(relative))
    return detail


def _confirmation_one(profile, replicate: int, root_seed: int) -> dict[str, object]:
    """Run the unchanged v1 LRT once while collecting scalar audit details.

    ``_lrt_one`` resolves ``fit_alternative_for_lrt`` from its module globals.
    We temporarily wrap that callable to observe returned scalar diagnostics.
    Publication execution uses process workers, never threads, so each worker
    has an isolated module namespace and this instrumentation cannot race.
    """

    original_fit: Callable = inference.fit_alternative_for_lrt
    call_details: list[dict[str, object]] = []
    fit_call_attempt_count = 0

    def recording_fit(values, **kwargs):
        nonlocal fit_call_attempt_count
        fit_call_attempt_count += 1
        fit, null_fit, statistic, raw = original_fit(values, **kwargs)
        fitting_raw = float(
            2.0
            * len(values)
            * (float(null_fit.average_nll) - float(fit.average_nll))
        )
        detail: dict[str, object] = {
            "fit_success": bool(fit.success),
            "null_success": bool(null_fit.success),
            "statistic": float(statistic),
            "raw": float(raw),
            "fitting_raw": fitting_raw,
        }
        if not call_details:
            detail.update(_fit_detail(fit, np.asarray(values, dtype=float)))
        call_details.append(detail)
        return fit, null_fit, statistic, raw

    inference.fit_alternative_for_lrt = recording_fit
    try:
        row = inference._lrt_one(
            profile,
            n=CONFIRMATION_N,
            separation=CONFIRMATION_SEPARATION,
            replicate=replicate,
            base_seed=root_seed,
        )
    finally:
        inference.fit_alternative_for_lrt = original_fit

    valid_bootstrap: list[dict[str, object]] = []
    for detail in call_details[1:]:
        if (
            detail["fit_success"]
            and detail["null_success"]
            and np.isfinite(detail["statistic"])
            and np.isfinite(detail["raw"])
            and np.isfinite(detail["fitting_raw"])
        ):
            valid_bootstrap.append(detail)
    if len(valid_bootstrap) != int(row["bootstrap_valid"]):
        raise RuntimeError("instrumented bootstrap-valid count disagrees with _lrt_one")

    boot_stat = np.asarray(
        [detail["statistic"] for detail in valid_bootstrap], dtype=float
    )
    boot_raw = np.asarray([detail["raw"] for detail in valid_bootstrap], dtype=float)
    observed_statistic = float(row["statistic"])
    exceedance_count: int | float = np.nan
    q90 = q95 = q99 = np.nan
    zero_count = fallback_count = material_fallback_count = 0
    if boot_stat.size:
        exceedance_count = int(np.sum(boot_stat >= observed_statistic))
        q90, q95, q99 = (
            float(value)
            for value in np.quantile(boot_stat, (0.90, 0.95, 0.99), method="linear")
        )
        zero_count = int(np.sum(boot_stat == 0.0))
        fallback_count = int(np.sum(boot_raw < 0.0))
        material_fallback_count = int(np.sum(boot_raw < -1e-7 * CONFIRMATION_N))
        reconstructed_p = (1.0 + exceedance_count) / (boot_stat.size + 1.0)
        if bool(row["analysis_success"]) and not np.isclose(
            reconstructed_p,
            float(row["bootstrap_pvalue"]),
            rtol=0.0,
            atol=2e-15,
        ):
            raise RuntimeError("instrumented exceedance count disagrees with p-value")

    observed_detail = call_details[0] if call_details else {}
    row.update(
        {
            "experiment": "lrt_null_confirmation",
            "confirmation_protocol_version": CONFIRMATION_PROTOCOL_VERSION,
            "confirmation_master_seed": CONFIRMATION_MASTER_SEED,
            "confirmation_replicate_root_seed": root_seed,
            "confirmation_bootstrap_parent_seed": inference._spawn_seed(
                root_seed, 13, CONFIRMATION_N, 0, replicate
            ),
            "confirmation_analysis_valid_95": bool(
                row["analysis_success"]
                and int(row["bootstrap_valid"])
                >= minimum_confirmation_bootstrap_valid(profile)
            ),
            "instrumented_fit_call_attempt_count": fit_call_attempt_count,
            "instrumented_fit_return_count": len(call_details),
            "instrumented_bootstrap_fit_return_count": max(
                0, len(call_details) - 1
            ),
            "bootstrap_exceedance_count": exceedance_count,
            "bootstrap_empirical_quantile_90": q90,
            "bootstrap_empirical_quantile_95": q95,
            "bootstrap_empirical_quantile_99": q99,
            "bootstrap_quantile_method": "numpy_linear",
            "bootstrap_statistic_zero_count": zero_count,
            "bootstrap_statistic_zero_fraction": (
                float(zero_count / boot_stat.size) if boot_stat.size else np.nan
            ),
            "bootstrap_endpoint_fallback_count": fallback_count,
            "bootstrap_material_endpoint_fallback_count": material_fallback_count,
            "observed_statistic_minus_raw_statistic": (
                float(row["statistic"] - row["raw_statistic"])
                if np.isfinite(row["statistic"]) and np.isfinite(row["raw_statistic"])
                else np.nan
            ),
            **{
                name: observed_detail.get(name, np.nan)
                for name in REQUIRED_CONFIRMATION_ROW_COLUMNS
                if name.startswith("fitted_alternative_")
            },
        }
    )
    return row


def _json_clean(value):
    return inference._json_clean(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        Path(inference.__file__).resolve(),
        ROOT / "simulations" / "wfcm_core.py",
        ROOT / "simulations" / "__init__.py",
        ROOT / "simulations" / "tests" / "test_lrt_null_confirmation.py",
        ROOT / "revision_work" / "lrt_null_confirmation_preregistration.md",
        ROOT / "revision_work" / "lrt_null_confirmation_writeback_template.tex",
        FROZEN_SOURCE_MANIFEST,
    )


def source_manifest_entries() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256_file(path)
        for path in source_files()
    }


def write_source_manifest(output: Path) -> dict[str, str]:
    entries = source_manifest_entries()
    text = "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    (output / "source_manifest.sha256").write_text(text, encoding="utf-8")
    return entries


def _read_sha256_manifest(path: Path) -> dict[str, str]:
    recorded: dict[str, str] = {}
    if not path.exists():
        return recorded
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    return recorded


def validate_frozen_source_manifest() -> dict[str, object]:
    """Verify the independently frozen prelaunch source bundle."""

    expected = _read_sha256_manifest(FROZEN_SOURCE_MANIFEST)
    current: dict[str, str | None] = {}
    safe_names = True
    for name in expected:
        candidate = (ROOT / name).resolve()
        if candidate != ROOT and ROOT not in candidate.parents:
            safe_names = False
            current[name] = None
        else:
            current[name] = _sha256_file(candidate) if candidate.is_file() else None
    checks = {
        "frozen_manifest_present": FROZEN_SOURCE_MANIFEST.is_file(),
        "frozen_manifest_nonempty": bool(expected),
        "frozen_manifest_exact_file_set": set(expected) == set(FROZEN_SOURCE_NAMES),
        "frozen_manifest_digest_schema": bool(expected)
        and all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in expected.values()
        ),
        "frozen_manifest_paths_within_project": safe_names,
        "frozen_source_hashes_match": bool(expected) and current == expected,
    }
    return {
        "checks": checks,
        "all_frozen_source_checks_passed": all(checks.values()),
        "frozen_manifest_sha256": (
            _sha256_file(FROZEN_SOURCE_MANIFEST)
            if FROZEN_SOURCE_MANIFEST.is_file()
            else None
        ),
        "expected_source_sha256": expected,
        "current_source_sha256": current,
    }


def validate_source_manifest(output: Path) -> dict[str, object]:
    manifest = output / "source_manifest.sha256"
    recorded = _read_sha256_manifest(manifest)
    current = source_manifest_entries()
    return {
        "source_manifest_present": manifest.exists(),
        "source_manifest_matches_current_code": recorded == current,
        "recorded_source_manifest": recorded,
        "current_source_manifest": current,
    }


def directory_fingerprints(directory: Path) -> dict[str, str]:
    """Hash every regular file in a result directory without mutating it."""

    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): _sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def validate_primary_run(primary_dir: Path = PRIMARY_LRT_OUTPUT) -> dict[str, object]:
    config_path = primary_dir / "run_config.json"
    raw_path = primary_dir / "lrt_raw.csv"
    if not config_path.exists() or not raw_path.exists():
        raise RuntimeError("the immutable primary LRT result is absent")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(raw_path)
    profile = config.get("profile", {})
    pinned_artifact_matches = {
        name: (primary_dir / name).exists()
        and _sha256_file(primary_dir / name) == expected
        for name, expected in PRIMARY_PINNED_ARTIFACT_SHA256.items()
    }
    null_x, null_valid, null_attempted = primary_n500_null_counts(frame)
    checks = {
        "primary_experiment": config.get("experiment") == "lrt",
        "primary_seed": config.get("seed") == PRIMARY_LRT_BASE_SEED,
        "primary_protocol": config.get("lrt_protocol_version")
        == inference.LRT_PROTOCOL_VERSION,
        "primary_B": profile.get("lrt_bootstrap_reps")
        == PRIMARY_LRT_BOOTSTRAP_REPS,
        "primary_starts": profile.get("fit_n_init")
        == CONFIRMATION_ALTERNATIVE_N_INIT,
        "primary_maxiter": profile.get("maxiter") == CONFIRMATION_MAXITER,
        "primary_n_grid": tuple(profile.get("lrt_ns", ())) == PRIMARY_LRT_NS,
        "primary_R_per_cell": profile.get("lrt_reps") == PRIMARY_LRT_REPS,
        "primary_row_count": len(frame)
        == len(PRIMARY_LRT_NS)
        * len(PRIMARY_LRT_SEPARATIONS)
        * PRIMARY_LRT_REPS,
        "primary_unique_design_keys": not frame.duplicated(
            ["n", "separation", "replicate"]
        ).any(),
        "primary_protocol_rows": "lrt_protocol_version" in frame
        and frame["lrt_protocol_version"].eq(inference.LRT_PROTOCOL_VERSION).all(),
        "primary_pinned_artifact_hashes": all(pinned_artifact_matches.values()),
        "primary_n500_null_result": (
            null_x == 14 and null_valid == 150 and null_attempted == 150
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError("primary LRT integrity gate failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "pinned_artifact_matches": pinned_artifact_matches,
        "file_sha256": directory_fingerprints(primary_dir),
    }


def assert_output_safe(
    output: Path,
    *,
    primary_dir: Path = PRIMARY_LRT_OUTPUT,
    profile_name: str | None = None,
) -> None:
    output_resolved = output.resolve()
    primary_resolved = primary_dir.resolve()
    paths_overlap = bool(
        output_resolved == primary_resolved
        or primary_resolved in output_resolved.parents
        or output_resolved in primary_resolved.parents
    )
    if paths_overlap:
        raise ValueError(
            "confirmation output must not equal, contain, or be inside the "
            "primary LRT directory"
        )
    if profile_name == "full":
        if output_resolved != DEFAULT_OUTPUT.resolve():
            raise ValueError(
                "the full confirmation may run only at its canonical output path"
            )
        # Even an empty pre-existing directory is ambiguous after a failed or
        # moved launch.  The one-time full run must create the path itself.
        if output.exists():
            raise FileExistsError(
                "the canonical full-confirmation output must be absent at launch"
            )
    elif profile_name == "smoke" and output_resolved == DEFAULT_OUTPUT.resolve():
        raise ValueError("a smoke run must not occupy the canonical full output path")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("confirmation output exists and is not a directory")
        primary_collisions = [
            name for name in PRIMARY_ARTIFACT_NAMES if (output / name).exists()
        ]
        confirmation_collisions = [
            name for name in CONFIRMATION_ARTIFACT_NAMES if (output / name).exists()
        ]
        if primary_collisions:
            raise FileExistsError(
                "refusing a directory containing primary-named artifacts: "
                + ", ".join(primary_collisions)
            )
        if confirmation_collisions:
            raise FileExistsError(
                "confirmation is immutable and cannot overwrite: "
                + ", ".join(confirmation_collisions)
            )
        if any(output.iterdir()):
            raise FileExistsError(
                "confirmation output must be absent or, for smoke only, empty"
            )


def _write_rows(rows: list[dict[str, object]], output: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{RAW_STEM}_raw.csv", index=False)
    with (output / f"{RAW_STEM}_raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_clean(row), sort_keys=True, allow_nan=False) + "\n"
            )
    return frame


def validate_confirmation_rows(
    frame: pd.DataFrame,
    *,
    profile_name: str,
    profile,
) -> dict[str, object]:
    required = set(REQUIRED_BASE_ROW_COLUMNS + REQUIRED_CONFIRMATION_ROW_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    reps = int(profile.lrt_reps)
    roots = confirmation_root_schedule(reps)
    expected_replicates = list(range(reps))
    replicate_numeric = (
        pd.to_numeric(frame["replicate"], errors="coerce")
        if "replicate" in frame
        else pd.Series(dtype=float)
    )
    replicate_indices_valid = bool(
        len(replicate_numeric) == len(frame)
        and replicate_numeric.notna().all()
        and np.allclose(
            replicate_numeric.to_numpy(dtype=float),
            np.round(replicate_numeric.to_numpy(dtype=float)),
            rtol=0.0,
            atol=0.0,
        )
        and (replicate_numeric >= 0).all()
        and (replicate_numeric < reps).all()
    )
    observed_replicates = (
        sorted(replicate_numeric.astype(int).tolist())
        if replicate_indices_valid
        else []
    )
    root_expected = (
        pd.Series(
            [roots[int(replicate)] for replicate in replicate_numeric],
            index=frame.index,
            dtype="object",
        )
        if not missing and len(frame) and replicate_indices_valid
        else pd.Series(dtype="object")
    )
    expected_data = pd.Series(
        [
            inference._spawn_seed(int(root), 11, CONFIRMATION_N, 0, int(replicate))
            for root, replicate in zip(root_expected, frame["replicate"], strict=True)
        ],
        index=frame.index,
    ) if len(root_expected) else pd.Series(dtype=float)
    expected_fit = pd.Series(
        [
            inference._spawn_seed(int(root), 12, CONFIRMATION_N, 0, int(replicate))
            for root, replicate in zip(root_expected, frame["replicate"], strict=True)
        ],
        index=frame.index,
    ) if len(root_expected) else pd.Series(dtype=float)
    expected_boot_parent = pd.Series(
        [
            inference._spawn_seed(int(root), 13, CONFIRMATION_N, 0, int(replicate))
            for root, replicate in zip(root_expected, frame["replicate"], strict=True)
        ],
        index=frame.index,
    ) if len(root_expected) else pd.Series(dtype=float)

    boolean_columns = (
        "analysis_success",
        "confirmation_analysis_valid_95",
        "fit_success",
        "null_fit_success",
        "null_optimizer_used",
        "scoring_adaptive_local_refinement",
    )
    boolean_encoding = bool(
        not missing
        and all(
            frame[column]
            .map(lambda value: isinstance(value, (bool, np.bool_)))
            .all()
            for column in boolean_columns
        )
    )

    def exact_boolean_series(column: str) -> pd.Series:
        if missing:
            return pd.Series(dtype=bool)
        return frame[column].map(
            lambda value: bool(value)
            if isinstance(value, (bool, np.bool_))
            else False
        )

    engine_success = exact_boolean_series("analysis_success")
    confirmation_success = exact_boolean_series(
        "confirmation_analysis_valid_95"
    )
    valid = frame[engine_success] if not missing else pd.DataFrame()
    p_identity = True
    reject_identity = True
    rejection_boolean_encoding = True
    quantile_order = True
    count_bounds = True
    count_integrality = True
    candidate_identity = True
    difference_identity = True
    endpoint_diagnostic_identities = True
    loglik_diagnostic_identities = True
    fraction_identities = True
    cross_objective_fraction_identities = True
    parameter_constraints = True
    ess_diagnostics = True
    analytic_null_diagnostics = True
    raw_gradient_diagnostics = True
    boundary_diagnostics = True
    projected_gradient_diagnostics = True
    confirmation_success_B95_requirement = True
    confirmation_validity_identity = True
    bootstrap_count_hierarchy = True
    instrumentation_call_identities = True
    if not missing:
        requested_all = pd.to_numeric(frame["bootstrap_requested"], errors="coerce")
        finite_all = pd.to_numeric(frame["bootstrap_finite"], errors="coerce")
        bvalid_all = pd.to_numeric(frame["bootstrap_valid"], errors="coerce")
        optimizer_all = pd.to_numeric(
            frame["bootstrap_optimizer_success"], errors="coerce"
        )
        hierarchy_values = pd.concat(
            [requested_all, finite_all, bvalid_all, optimizer_all], axis=1
        ).to_numpy(dtype=float)
        bootstrap_count_hierarchy = bool(
            np.isfinite(hierarchy_values).all()
            and np.allclose(
                hierarchy_values,
                np.round(hierarchy_values),
                rtol=0.0,
                atol=0.0,
            )
            and (requested_all >= 0).all()
            and (finite_all >= 0).all()
            and (bvalid_all >= 0).all()
            and (optimizer_all >= 0).all()
            and (optimizer_all == bvalid_all).all()
            and (bvalid_all <= finite_all).all()
            and (finite_all <= requested_all).all()
        )

        attempts = pd.to_numeric(
            frame["instrumented_fit_call_attempt_count"], errors="coerce"
        )
        returns = pd.to_numeric(
            frame["instrumented_fit_return_count"], errors="coerce"
        )
        bootstrap_returns = pd.to_numeric(
            frame["instrumented_bootstrap_fit_return_count"], errors="coerce"
        )
        call_values = pd.concat(
            [attempts, returns, bootstrap_returns], axis=1
        ).to_numpy(dtype=float)
        allowed_attempt_counts = attempts.isin(
            [0, 1, int(profile.lrt_bootstrap_reps) + 1]
        )
        instrumentation_call_identities = bool(
            np.isfinite(call_values).all()
            and np.allclose(
                call_values, np.round(call_values), rtol=0.0, atol=0.0
            )
            and (attempts >= 0).all()
            and (returns >= 0).all()
            and (bootstrap_returns >= 0).all()
            and allowed_attempt_counts.all()
            and (returns <= attempts).all()
            and (
                bootstrap_returns
                == np.maximum(returns - 1, 0)
            ).all()
            and (bootstrap_returns <= profile.lrt_bootstrap_reps).all()
            and (bvalid_all <= bootstrap_returns).all()
            and (
                attempts[engine_success]
                == int(profile.lrt_bootstrap_reps) + 1
            ).all()
        )
    if len(valid):
        expected_p = (
            1.0 + pd.to_numeric(valid["bootstrap_exceedance_count"])
        ) / (pd.to_numeric(valid["bootstrap_valid"]) + 1.0)
        p_identity = bool(
            np.allclose(
                pd.to_numeric(valid["bootstrap_pvalue"]),
                expected_p,
                rtol=0.0,
                atol=2e-15,
            )
        )
        rejection_boolean_encoding = bool(
            valid["bootstrap_reject_05"]
            .map(lambda value: isinstance(value, (bool, np.bool_)))
            .all()
        )
        reject_identity = bool(
            rejection_boolean_encoding
            and (
                valid["bootstrap_reject_05"].astype(bool)
                == (pd.to_numeric(valid["bootstrap_pvalue"]) <= 0.05)
            ).all()
        )
        q90 = pd.to_numeric(valid["bootstrap_empirical_quantile_90"])
        q95 = pd.to_numeric(valid["bootstrap_empirical_quantile_95"])
        q99 = pd.to_numeric(valid["bootstrap_empirical_quantile_99"])
        quantile_order = bool(((q90 <= q95) & (q95 <= q99)).all())
        bvalid = pd.to_numeric(valid["bootstrap_valid"])
        count_columns = (
            "bootstrap_exceedance_count",
            "bootstrap_statistic_zero_count",
            "bootstrap_endpoint_fallback_count",
            "bootstrap_material_endpoint_fallback_count",
        )
        count_bounds = all(
            bool(
                (
                    (pd.to_numeric(valid[column]) >= 0)
                    & (pd.to_numeric(valid[column]) <= bvalid)
                ).all()
            )
            for column in count_columns
        )
        count_integrality = all(
            bool(
                np.allclose(
                    pd.to_numeric(valid[column]),
                    np.round(pd.to_numeric(valid[column])),
                    rtol=0.0,
                    atol=0.0,
                )
            )
            for column in count_columns
        )
        raw = pd.to_numeric(valid["raw_statistic"])
        statistic = pd.to_numeric(valid["statistic"])
        candidate_identity = bool(
            np.allclose(statistic, np.maximum(0.0, raw), rtol=0.0, atol=2e-12)
        )
        difference_identity = bool(
            np.allclose(
                pd.to_numeric(valid["observed_statistic_minus_raw_statistic"]),
                statistic - raw,
                rtol=0.0,
                atol=2e-12,
            )
        )
        negative_tolerance = pd.to_numeric(valid["negative_raw_tolerance"])
        endpoint_diagnostic_identities = bool(
            (
                valid["scored_candidate_selected"]
                == np.where(
                    raw < 0.0,
                    "embedded_analytic_null",
                    "fitted_alternative",
                )
            ).all()
            and (
                valid["embedded_null_fallback_used"].astype(bool)
                == (raw < 0.0)
            ).all()
            and (
                valid["materially_negative_raw"].astype(bool)
                == (raw < -negative_tolerance)
            ).all()
            and np.allclose(
                pd.to_numeric(valid["negative_raw_magnitude"]),
                np.maximum(0.0, -raw),
                rtol=0.0,
                atol=2e-12,
            )
            and np.allclose(
                negative_tolerance,
                1e-7 * pd.to_numeric(valid["n"]),
                rtol=0.0,
                atol=2e-15,
            )
        )
        analytic_loglik = pd.to_numeric(valid["analytic_null_loglik"])
        fitted_loglik = pd.to_numeric(valid["fitted_alternative_loglik"])
        loglik_diagnostic_identities = bool(
            np.isfinite(analytic_loglik).all()
            and np.isfinite(fitted_loglik).all()
            and np.allclose(
                fitted_loglik,
                analytic_loglik + raw / 2.0,
                rtol=2e-13,
                atol=2e-12,
            )
        )
        zero_fraction = pd.to_numeric(valid["bootstrap_statistic_zero_fraction"])
        fallback_fraction = pd.to_numeric(valid["bootstrap_negative_raw_fraction"])
        material_fraction = pd.to_numeric(
            valid["bootstrap_materially_negative_raw_fraction"]
        )
        fraction_identities = bool(
            np.allclose(
                zero_fraction,
                pd.to_numeric(valid["bootstrap_statistic_zero_count"]) / bvalid,
                rtol=0.0,
                atol=2e-15,
            )
            and np.allclose(
                fallback_fraction,
                pd.to_numeric(valid["bootstrap_endpoint_fallback_count"]) / bvalid,
                rtol=0.0,
                atol=2e-15,
            )
            and np.allclose(
                material_fraction,
                pd.to_numeric(
                    valid["bootstrap_material_endpoint_fallback_count"]
                )
                / bvalid,
                rtol=0.0,
                atol=2e-15,
            )
        )
        cross_defined = pd.to_numeric(
            valid["bootstrap_cross_objective_diagnostic_defined"]
        )
        cross_count = pd.to_numeric(
            valid["bootstrap_cross_objective_fallback_count"]
        )
        material_cross_count = pd.to_numeric(
            valid["bootstrap_material_cross_objective_mismatch_count"]
        )
        cross_objective_fraction_identities = bool(
            np.allclose(cross_defined, bvalid, rtol=0.0, atol=0.0)
            and ((cross_count >= 0) & (cross_count <= cross_defined)).all()
            and (
                (material_cross_count >= 0)
                & (material_cross_count <= cross_defined)
            ).all()
            and np.allclose(
                pd.to_numeric(
                    valid["bootstrap_cross_objective_fallback_fraction"]
                ),
                cross_count / cross_defined,
                rtol=0.0,
                atol=2e-15,
            )
            and np.allclose(
                pd.to_numeric(
                    valid[
                        "bootstrap_material_cross_objective_mismatch_fraction"
                    ]
                ),
                material_cross_count / cross_defined,
                rtol=0.0,
                atol=2e-15,
            )
        )
        weights = valid[
            ["fitted_alternative_weight_1", "fitted_alternative_weight_2"]
        ].apply(pd.to_numeric)
        centers = valid[
            ["fitted_alternative_center_1", "fitted_alternative_center_2"]
        ].apply(pd.to_numeric)
        sigma = pd.to_numeric(valid["fitted_alternative_sigma"])
        fitted_m = pd.to_numeric(valid["fitted_alternative_m"])
        raw_gradient = pd.to_numeric(
            valid["fitted_alternative_raw_gradient_inf_norm"]
        )
        parameter_constraints = bool(
            np.isfinite(sigma).all()
            and (sigma > 0.0).all()
            and np.isfinite(centers.to_numpy()).all()
            and np.isfinite(weights.to_numpy()).all()
            and (weights.to_numpy() > 0.0).all()
            and np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2e-12)
            and np.allclose(fitted_m, 2.0, rtol=0.0, atol=2e-12)
        )
        ess = pd.to_numeric(valid["ess_fraction"])
        ess_diagnostics = bool(
            np.isfinite(ess).all() and (ess > 0.0).all() and (ess <= 1.0).all()
        )
        null_mean = pd.to_numeric(valid["analytic_null_mean"])
        null_normal_sd = pd.to_numeric(valid["analytic_null_normal_sd"])
        null_wfcm_sigma = pd.to_numeric(valid["analytic_null_wfcm_sigma"])
        null_gradient = pd.to_numeric(valid["null_gradient_inf_norm"])
        analytic_null_diagnostics = bool(
            np.isfinite(null_mean).all()
            and np.isfinite(null_normal_sd).all()
            and (null_normal_sd > 0.0).all()
            and np.isfinite(null_wfcm_sigma).all()
            and np.allclose(
                null_wfcm_sigma,
                np.sqrt(2.0) * null_normal_sd,
                rtol=2e-13,
                atol=2e-13,
            )
            and np.allclose(null_gradient, 0.0, rtol=0.0, atol=0.0)
        )
        raw_gradient_diagnostics = bool(
            np.isfinite(raw_gradient).all() and (raw_gradient >= 0.0).all()
        )
        projected_gradient = pd.to_numeric(valid["gradient_inf_norm"])
        projected_gradient_diagnostics = bool(
            np.isfinite(projected_gradient).all()
            and (projected_gradient >= 0.0).all()
        )
        absolute_margin = pd.to_numeric(
            valid["fitted_alternative_min_free_distance_to_bound"]
        )
        relative_margin = pd.to_numeric(
            valid["fitted_alternative_min_relative_free_distance_to_bound"]
        )
        boundary_diagnostics = bool(
            np.isfinite(absolute_margin).all()
            and np.isfinite(relative_margin).all()
            and (absolute_margin >= -2e-12).all()
            and (relative_margin >= -2e-12).all()
            and (relative_margin <= 0.5 + 2e-12).all()
        )
    confirmation_valid_rows = (
        frame[confirmation_success] if not missing else pd.DataFrame()
    )
    if len(confirmation_valid_rows):
        confirmation_success_B95_requirement = bool(
            (
                pd.to_numeric(confirmation_valid_rows["bootstrap_valid"])
                >= minimum_confirmation_bootstrap_valid(profile)
            ).all()
        )
    if not missing:
        expected_confirmation_success = engine_success & (
            pd.to_numeric(frame["bootstrap_valid"], errors="coerce")
            >= minimum_confirmation_bootstrap_valid(profile)
        )
        confirmation_validity_identity = bool(
            confirmation_success.equals(expected_confirmation_success)
        )

    checks = {
        "required_row_schema": not missing,
        "boolean_field_encoding": boolean_encoding,
        "experiment_label": frame["experiment"].eq(
            "lrt_null_confirmation"
        ).all() if not missing else False,
        "row_count": len(frame) == reps,
        "unique_design_keys": (
            not frame.duplicated(["n", "separation", "replicate"]).any()
            if not missing
            else False
        ),
        "complete_replicate_grid": observed_replicates == expected_replicates,
        "sample_size": frame["n"].eq(CONFIRMATION_N).all() if not missing else False,
        "null_only": frame["separation"].eq(0.0).all() if not missing else False,
        "confirmation_protocol": frame["confirmation_protocol_version"].eq(
            CONFIRMATION_PROTOCOL_VERSION
        ).all() if not missing else False,
        "confirmation_master_seed": frame["confirmation_master_seed"].eq(
            CONFIRMATION_MASTER_SEED
        ).all() if not missing else False,
        "lrt_protocol": frame["lrt_protocol_version"].eq(
            inference.LRT_PROTOCOL_VERSION
        ).all() if not missing else False,
        "statistic_name": frame["statistic_name"].eq(
            inference.PRIMARY_LRT_NAME
        ).all() if not missing else False,
        "B": frame["bootstrap_requested"].eq(profile.lrt_bootstrap_reps).all()
        if not missing
        else False,
        "alternative_starts": frame["alternative_n_init"].eq(
            profile.fit_n_init
        ).all() if not missing else False,
        "null_starts_zero": frame["null_n_init"].eq(0).all()
        if not missing
        else False,
        "importance_size": frame["importance_size"].eq(
            CONFIRMATION_IMPORTANCE_SIZE
        ).all() if not missing else False,
        "quadrature": (
            frame["scoring_method"].eq("deterministic_1d_quadrature").all()
            and frame["scoring_grid_size"].eq(100_001).all()
            and frame["scoring_base_grid_size"].eq(100_001).all()
            and frame["scoring_adaptive_local_refinement"].eq(True).all()
            and frame["scoring_tail_sd_multiplier"].eq(12.0).all()
        ) if not missing else False,
        "analytic_null": (
            frame["null_estimator"].eq(inference.LRT_NULL_ESTIMATOR).all()
            and frame["null_scoring_method"].eq(
                inference.LRT_NULL_SCORING_METHOD
            ).all()
            and (~frame["null_optimizer_used"].astype(bool)).all()
        ) if not missing else False,
        "candidate_max_rule": frame["alternative_candidate_rule"].eq(
            inference.LRT_ALTERNATIVE_CANDIDATE_RULE
        ).all() if not missing else False,
        "root_seed_formula": bool(
            len(root_expected)
            and np.array_equal(
                frame["confirmation_replicate_root_seed"].astype(object).to_numpy(),
                root_expected.to_numpy(),
            )
        ) if not missing else False,
        "data_seed_formula": bool(
            len(expected_data)
            and np.array_equal(
                pd.to_numeric(frame["data_seed"]).to_numpy(), expected_data.to_numpy()
            )
        ) if not missing else False,
        "fit_seed_formula": bool(
            len(expected_fit)
            and np.array_equal(
                pd.to_numeric(frame["fit_seed"]).to_numpy(), expected_fit.to_numpy()
            )
        ) if not missing else False,
        "bootstrap_parent_seed_formula": bool(
            len(expected_boot_parent)
            and np.array_equal(
                pd.to_numeric(frame["confirmation_bootstrap_parent_seed"]).to_numpy(),
                expected_boot_parent.to_numpy(),
            )
        ) if not missing else False,
        "bootstrap_quantile_method": frame["bootstrap_quantile_method"].eq(
            "numpy_linear"
        ).all() if not missing else False,
        "bootstrap_count_hierarchy": bootstrap_count_hierarchy,
        "instrumentation_call_identities": instrumentation_call_identities,
        "plus_one_pvalue_identity": p_identity,
        "rejection_boolean_encoding": rejection_boolean_encoding,
        "rejection_identity": reject_identity,
        "empirical_quantile_order": quantile_order,
        "bootstrap_count_bounds": count_bounds,
        "bootstrap_count_integrality": count_integrality,
        "candidate_max_statistic_identity": candidate_identity,
        "observed_statistic_raw_difference_identity": difference_identity,
        "endpoint_diagnostic_identities": endpoint_diagnostic_identities,
        "loglik_diagnostic_identities": loglik_diagnostic_identities,
        "bootstrap_fraction_count_identities": fraction_identities,
        "bootstrap_cross_objective_fraction_identities": (
            cross_objective_fraction_identities
        ),
        "fitted_alternative_parameter_constraints": parameter_constraints,
        "fitted_alternative_ess_diagnostics": ess_diagnostics,
        "analytic_null_diagnostics": analytic_null_diagnostics,
        "fitted_alternative_projected_gradient_diagnostics": (
            projected_gradient_diagnostics
        ),
        "fitted_alternative_raw_gradient_diagnostics": raw_gradient_diagnostics,
        "fitted_alternative_boundary_diagnostics": boundary_diagnostics,
        "confirmation-valid_rows_meet_95pct_bootstrap_rule": (
            confirmation_success_B95_requirement
        ),
        "confirmation_validity_indicator_identity": confirmation_validity_identity,
        "publication_profile_name": profile_name == "full",
    }
    if profile_name == "full":
        checks["all_outer_analyses_valid_for_fixed_R_interpretation"] = bool(
            len(frame) == CONFIRMATION_REPS and confirmation_success.all()
        )
    return {
        "checks": checks,
        "missing_columns": missing,
        "all_structural_checks_passed": all(
            value
            for key, value in checks.items()
            if key
            not in {
                "publication_profile_name",
                "all_outer_analyses_valid_for_fixed_R_interpretation",
            }
        ),
    }


def validate_serializations(output: Path) -> dict[str, object]:
    """Cross-check CSV and strict JSONL on all scalar audit columns."""

    csv_path = output / f"{RAW_STEM}_raw.csv"
    jsonl_path = output / f"{RAW_STEM}_raw.jsonl"
    if not csv_path.exists() or not jsonl_path.exists():
        return {
            "csv_present": csv_path.exists(),
            "jsonl_present": jsonl_path.exists(),
            "row_counts_match": False,
            "required_scalar_values_match": False,
        }
    csv_frame = pd.read_csv(csv_path)
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    json_frame = pd.DataFrame(records)
    columns = list(REQUIRED_BASE_ROW_COLUMNS + REQUIRED_CONFIRMATION_ROW_COLUMNS)
    match = len(csv_frame) == len(json_frame) and set(columns).issubset(
        csv_frame.columns
    ) and set(columns).issubset(json_frame.columns)
    if match:
        for column in columns:
            csv_values = csv_frame[column]
            json_values = json_frame[column]
            csv_numeric = pd.to_numeric(csv_values, errors="coerce")
            json_numeric = pd.to_numeric(json_values, errors="coerce")
            both_numeric_or_missing = (
                (csv_numeric.notna() | csv_values.isna())
                & (json_numeric.notna() | json_values.isna())
            ).all()
            if both_numeric_or_missing:
                if not np.allclose(
                    csv_numeric.to_numpy(dtype=float),
                    json_numeric.to_numpy(dtype=float),
                    rtol=2e-13,
                    atol=2e-13,
                    equal_nan=True,
                ):
                    match = False
                    break
            else:
                csv_text = csv_values.fillna("<NULL>").astype(str)
                json_text = json_values.fillna("<NULL>").astype(str)
                if not csv_text.equals(json_text):
                    match = False
                    break
    return {
        "csv_present": True,
        "jsonl_present": True,
        "csv_rows": int(len(csv_frame)),
        "jsonl_rows": int(len(json_frame)),
        "row_counts_match": len(csv_frame) == len(json_frame),
        "required_scalar_values_match": bool(match),
    }


def validate_derived_outputs(
    output: Path,
    *,
    profile_name: str,
    primary_frame: pd.DataFrame | None,
) -> dict[str, object]:
    """Recompute the summary/write-back from stored raw rows and compare."""

    raw_path = output / f"{RAW_STEM}_raw.csv"
    summary_path = output / f"{RAW_STEM}_summary.csv"
    writeback_path = output / "writeback_fields.json"
    present = {
        "raw_csv_present": raw_path.is_file(),
        "summary_csv_present": summary_path.is_file(),
        "writeback_json_present": writeback_path.is_file(),
    }
    if not all(present.values()):
        return {
            **present,
            "summary_matches_recomputed_raw": False,
            "writeback_matches_recomputed_summary": False,
        }

    raw_frame = pd.read_csv(raw_path)
    expected_summary = summarize_confirmation(
        raw_frame, profile_name, primary_frame=primary_frame
    )
    stored_summary = pd.read_csv(summary_path)
    summary_match = bool(
        len(stored_summary) == len(expected_summary)
        and list(stored_summary.columns) == list(expected_summary.columns)
    )
    if summary_match:
        for column in expected_summary.columns:
            expected_values = expected_summary[column]
            stored_values = stored_summary[column]
            expected_numeric = pd.to_numeric(expected_values, errors="coerce")
            stored_numeric = pd.to_numeric(stored_values, errors="coerce")
            numeric_or_missing = bool(
                ((expected_numeric.notna() | expected_values.isna())
                & (stored_numeric.notna() | stored_values.isna())).all()
            )
            if numeric_or_missing:
                if not np.allclose(
                    expected_numeric.to_numpy(dtype=float),
                    stored_numeric.to_numpy(dtype=float),
                    rtol=2e-13,
                    atol=2e-13,
                    equal_nan=True,
                ):
                    summary_match = False
                    break
            elif not expected_values.fillna("<NULL>").astype(str).equals(
                stored_values.fillna("<NULL>").astype(str)
            ):
                summary_match = False
                break

    expected_writeback = _json_clean(writeback_fields(expected_summary.iloc[0]))
    stored_writeback = json.loads(writeback_path.read_text(encoding="utf-8"))
    writeback_match = stored_writeback == expected_writeback
    return {
        **present,
        "summary_matches_recomputed_raw": summary_match,
        "writeback_matches_recomputed_summary": writeback_match,
    }


def binomial_intervals(
    successes: int, trials: int, *, confidence: float = 0.95
) -> dict[str, float]:
    """Return two-sided Clopper--Pearson and Wilson score intervals."""

    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials < 1
        or not 0 <= successes <= trials
    ):
        raise ValueError("successes and trials must be integers with 0 <= x <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    alpha = 1.0 - confidence
    cp_lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    cp_upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    z = float(norm.ppf(1.0 - alpha / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return {
        "clopper_pearson_lower": cp_lower,
        "clopper_pearson_upper": cp_upper,
        "wilson_lower": float(center - half_width),
        "wilson_upper": float(center + half_width),
    }


def primary_n500_null_counts(primary_frame: pd.DataFrame) -> tuple[int, int, int]:
    cell = primary_frame[
        (pd.to_numeric(primary_frame["n"], errors="coerce") == CONFIRMATION_N)
        & (
            pd.to_numeric(primary_frame["separation"], errors="coerce")
            == CONFIRMATION_SEPARATION
        )
    ]
    valid = cell[
        cell["analysis_success"].fillna(False).astype(bool)
        & (
            pd.to_numeric(cell["bootstrap_valid"], errors="coerce")
            >= CONFIRMATION_MIN_BOOTSTRAP_VALID
        )
    ]
    return (
        int(valid["bootstrap_reject_05"].astype(bool).sum()),
        int(len(valid)),
        int(len(cell)),
    )


def summarize_confirmation(
    frame: pd.DataFrame,
    profile_name: str,
    *,
    primary_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if profile_name == "full":
        minimum_valid = CONFIRMATION_MIN_BOOTSTRAP_VALID
    else:
        requested = int(pd.to_numeric(frame["bootstrap_requested"]).max())
        minimum_valid = int(
            np.ceil(CONFIRMATION_MIN_BOOTSTRAP_VALID_FRACTION * requested)
        )
    success = frame["analysis_success"].fillna(False).astype(bool) & (
        pd.to_numeric(frame["bootstrap_valid"], errors="coerce") >= minimum_valid
    )
    valid = frame[success]
    x = int(valid["bootstrap_reject_05"].astype(bool).sum())
    attempted = int(len(frame))
    valid_count = int(len(valid))
    eligible = bool(
        profile_name == "full"
        and attempted == CONFIRMATION_REPS
        and valid_count == CONFIRMATION_REPS
    )
    rate = x / valid_count if valid_count else np.nan
    mcse = np.sqrt(rate * (1.0 - rate) / valid_count) if valid_count else np.nan
    classification = classify_confirmation_count(x) if eligible else "not_eligible"
    confirmation_intervals = (
        binomial_intervals(x, valid_count) if valid_count else {
            "clopper_pearson_lower": np.nan,
            "clopper_pearson_upper": np.nan,
            "wilson_lower": np.nan,
            "wilson_upper": np.nan,
        }
    )
    primary_x = primary_n = primary_attempted = 0
    if primary_frame is not None:
        primary_x, primary_n, primary_attempted = primary_n500_null_counts(
            primary_frame
        )
    primary_intervals = (
        binomial_intervals(primary_x, primary_n) if primary_n else {
            "clopper_pearson_lower": np.nan,
            "clopper_pearson_upper": np.nan,
            "wilson_lower": np.nan,
            "wilson_upper": np.nan,
        }
    )
    pooled_eligible = eligible and primary_n == PRIMARY_LRT_REPS
    pooled_x = primary_x + x if pooled_eligible else 0
    pooled_n = primary_n + valid_count if pooled_eligible else 0
    pooled_attempted = (
        primary_attempted + attempted if pooled_eligible else 0
    )
    pooled_intervals = (
        binomial_intervals(pooled_x, pooled_n) if pooled_eligible else {
            "clopper_pearson_lower": np.nan,
            "clopper_pearson_upper": np.nan,
            "wilson_lower": np.nan,
            "wilson_upper": np.nan,
        }
    )
    return pd.DataFrame(
        [
            {
                "n": CONFIRMATION_N,
                "separation": CONFIRMATION_SEPARATION,
                "attempted_replicates": attempted,
                "analysis_valid_replicates": valid_count,
                "failed_replicates": attempted - valid_count,
                "bootstrap_rejection_count": x,
                "bootstrap_rejection_rate_analysis_valid": rate,
                "monte_carlo_standard_error": mcse,
                "confirmation_clopper_pearson_lower": confirmation_intervals[
                    "clopper_pearson_lower"
                ],
                "confirmation_clopper_pearson_upper": confirmation_intervals[
                    "clopper_pearson_upper"
                ],
                "confirmation_wilson_lower": confirmation_intervals["wilson_lower"],
                "confirmation_wilson_upper": confirmation_intervals["wilson_upper"],
                "exact_binomial_upper_tail_at_0.05": (
                    float(binom.sf(x - 1, CONFIRMATION_REPS, 0.05))
                    if eligible
                    else np.nan
                ),
                "fixed_R_interpretation_eligible": eligible,
                "prespecified_classification": classification,
                "primary_result_replaced": False,
                "pooled_with_primary_for_decision": False,
                "sequential_additions_permitted": False,
                "primary_n500_null_rejection_count": (
                    primary_x if primary_n else np.nan
                ),
                "primary_n500_null_analysis_valid_replicates": (
                    primary_n if primary_n else np.nan
                ),
                "primary_n500_null_attempted_replicates": (
                    primary_attempted if primary_attempted else np.nan
                ),
                "primary_n500_null_failed_replicates": (
                    primary_attempted - primary_n if primary_attempted else np.nan
                ),
                "primary_n500_null_rejection_rate": (
                    primary_x / primary_n if primary_n else np.nan
                ),
                "primary_clopper_pearson_lower": primary_intervals[
                    "clopper_pearson_lower"
                ],
                "primary_clopper_pearson_upper": primary_intervals[
                    "clopper_pearson_upper"
                ],
                "primary_wilson_lower": primary_intervals["wilson_lower"],
                "primary_wilson_upper": primary_intervals["wilson_upper"],
                "secondary_pooled_rejection_count": (
                    pooled_x if pooled_eligible else np.nan
                ),
                "secondary_pooled_analysis_valid_replicates": (
                    pooled_n if pooled_eligible else np.nan
                ),
                "secondary_pooled_attempted_replicates": (
                    pooled_attempted if pooled_eligible else np.nan
                ),
                "secondary_pooled_failed_replicates": (
                    pooled_attempted - pooled_n if pooled_eligible else np.nan
                ),
                "secondary_pooled_rejection_rate": (
                    pooled_x / pooled_n if pooled_eligible else np.nan
                ),
                "secondary_pooled_clopper_pearson_lower": pooled_intervals[
                    "clopper_pearson_lower"
                ],
                "secondary_pooled_clopper_pearson_upper": pooled_intervals[
                    "clopper_pearson_upper"
                ],
                "secondary_pooled_wilson_lower": pooled_intervals["wilson_lower"],
                "secondary_pooled_wilson_upper": pooled_intervals["wilson_upper"],
                "secondary_pooled_status": (
                    "descriptive_only_not_used_for_prespecified_decision"
                    if pooled_eligible
                    else "not_available"
                ),
            }
        ]
    )


def writeback_fields(summary: pd.Series) -> dict[str, object]:
    """Create exact, explicitly separated fields for later manuscript use."""

    def percent(value) -> str:
        return "NA" if pd.isna(value) else f"{100.0 * float(value):.2f}\\%"

    def interval(prefix: str) -> dict[str, object]:
        return {
            "confidence_level": 0.95,
            "clopper_pearson_lower": _json_clean(
                summary[prefix + "_clopper_pearson_lower"]
            ),
            "clopper_pearson_upper": _json_clean(
                summary[prefix + "_clopper_pearson_upper"]
            ),
            "wilson_lower": _json_clean(summary[prefix + "_wilson_lower"]),
            "wilson_upper": _json_clean(summary[prefix + "_wilson_upper"]),
            "clopper_pearson_95": (
                f"[{percent(summary[prefix + '_clopper_pearson_lower'])}, "
                f"{percent(summary[prefix + '_clopper_pearson_upper'])}]"
            ),
            "wilson_95": (
                f"[{percent(summary[prefix + '_wilson_lower'])}, "
                f"{percent(summary[prefix + '_wilson_upper'])}]"
            ),
        }

    return {
        "primary": {
            "label": "original primary n=500 null cell (retained)",
            "rejections": _json_clean(summary["primary_n500_null_rejection_count"]),
            "attempted": _json_clean(
                summary["primary_n500_null_attempted_replicates"]
            ),
            "analysis_valid": _json_clean(
                summary["primary_n500_null_analysis_valid_replicates"]
            ),
            "failed": _json_clean(
                summary["primary_n500_null_failed_replicates"]
            ),
            "denominator": _json_clean(
                summary["primary_n500_null_analysis_valid_replicates"]
            ),
            "denominator_kind": "analysis_valid",
            "rate_decimal": _json_clean(
                summary["primary_n500_null_rejection_rate"]
            ),
            "rate": percent(summary["primary_n500_null_rejection_rate"]),
            **interval("primary"),
        },
        "confirmation": {
            "label": "independent prespecified R=600 confirmation",
            "rejections": int(summary["bootstrap_rejection_count"]),
            "attempted": int(summary["attempted_replicates"]),
            "analysis_valid": int(summary["analysis_valid_replicates"]),
            "failed": int(summary["failed_replicates"]),
            "denominator": int(summary["analysis_valid_replicates"]),
            "denominator_kind": "analysis_valid",
            "rate_decimal": _json_clean(
                summary["bootstrap_rejection_rate_analysis_valid"]
            ),
            "rate": percent(summary["bootstrap_rejection_rate_analysis_valid"]),
            "monte_carlo_standard_error_decimal": _json_clean(
                summary["monte_carlo_standard_error"]
            ),
            "monte_carlo_standard_error": percent(
                summary["monte_carlo_standard_error"]
            ),
            **interval("confirmation"),
            "threshold_outcome": summary["prespecified_classification"],
            "fixed_thresholds": "X<=34; X=35--39; X>=40",
            "fixed_R_interpretation_eligible": bool(
                summary["fixed_R_interpretation_eligible"]
            ),
            "exact_binomial_upper_tail_at_0.05": _json_clean(
                summary["exact_binomial_upper_tail_at_0.05"]
            ),
        },
        "secondary_pooled": {
            "label": "secondary descriptive pooled estimate; not used for decision",
            "rejections": _json_clean(summary["secondary_pooled_rejection_count"]),
            "attempted": _json_clean(
                summary["secondary_pooled_attempted_replicates"]
            ),
            "analysis_valid": _json_clean(
                summary["secondary_pooled_analysis_valid_replicates"]
            ),
            "failed": _json_clean(
                summary["secondary_pooled_failed_replicates"]
            ),
            "denominator": _json_clean(
                summary["secondary_pooled_analysis_valid_replicates"]
            ),
            "denominator_kind": "analysis_valid",
            "rate_decimal": _json_clean(
                summary["secondary_pooled_rejection_rate"]
            ),
            "rate": percent(summary["secondary_pooled_rejection_rate"]),
            **interval("secondary_pooled"),
            "status": summary["secondary_pooled_status"],
        },
        "guardrails": {
            "primary_replaced": False,
            "pooled_used_for_prespecified_decision": False,
            "sequential_additions_permitted": False,
        },
    }


def _write_run_config(output: Path, profile_name: str, profile, jobs: int) -> dict:
    config = {
        "experiment": "lrt_null_confirmation",
        "profile_name": profile_name,
        "profile": asdict(profile),
        "jobs": jobs,
        "parallel_backend": "joblib_loky_processes",
        "seed": CONFIRMATION_MASTER_SEED,
        "seed_schedule": "smallest_collision_free_replicate_root-v1",
        "confirmation_protocol_version": CONFIRMATION_PROTOCOL_VERSION,
        "confirmation_n": CONFIRMATION_N,
        "confirmation_separation": CONFIRMATION_SEPARATION,
        "confirmation_reps": profile.lrt_reps,
        "confirmation_bootstrap_reps": profile.lrt_bootstrap_reps,
        "confirmation_alternative_n_init_total": profile.fit_n_init,
        "confirmation_maxiter": profile.maxiter,
        "confirmation_importance_size": CONFIRMATION_IMPORTANCE_SIZE,
        "bootstrap_quantile_method": "numpy_linear",
        "confirmation_minimum_bootstrap_valid_fraction": (
            CONFIRMATION_MIN_BOOTSTRAP_VALID_FRACTION
        ),
        "confirmation_minimum_bootstrap_valid": (
            minimum_confirmation_bootstrap_valid(profile)
        ),
        "lrt_protocol_version": inference.LRT_PROTOCOL_VERSION,
        "lrt_null_estimator": inference.LRT_NULL_ESTIMATOR,
        "lrt_null_scoring_method": inference.LRT_NULL_SCORING_METHOD,
        "lrt_alternative_candidate_rule": inference.LRT_ALTERNATIVE_CANDIDATE_RULE,
        "lrt_engine_sha256": lrt_engine_fingerprint(),
        "frozen_source_manifest_sha256": (
            _sha256_file(FROZEN_SOURCE_MANIFEST)
            if FROZEN_SOURCE_MANIFEST.is_file()
            else None
        ),
        "interpretation_thresholds": {
            "does_not_support_elevation": "X <= 34",
            "inconclusive": "35 <= X <= 39",
            "supports_elevation": "X >= 40",
        },
        "interpretation_requires_all_600_outer_analyses_valid": True,
        "primary_result_replaced": False,
        "pool_with_primary_for_prespecified_decision": False,
        "sequential_additions_permitted": False,
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    (output / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    return config


def run_confirmation(
    *,
    profile_name: str,
    output: Path,
    jobs: int,
    primary_dir: Path = PRIMARY_LRT_OUTPUT,
    require_primary: bool = True,
) -> dict[str, object]:
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if profile_name not in {"full", "smoke"}:
        raise ValueError("profile_name must be 'full' or 'smoke'")
    if profile_name == "full" and not require_primary:
        raise ValueError("the full confirmation requires the immutable primary gate")
    if (
        profile_name == "full"
        and primary_dir.resolve() != PRIMARY_LRT_OUTPUT.resolve()
    ):
        raise ValueError("the full confirmation requires the canonical primary path")
    profile = publication_profile() if profile_name == "full" else smoke_profile()
    validate_locked_engine(require_pinned_hash=profile_name == "full")
    frozen_source_audit = validate_frozen_source_manifest()
    if profile_name == "full" and not frozen_source_audit[
        "all_frozen_source_checks_passed"
    ]:
        raise RuntimeError("the frozen prelaunch source manifest does not match")
    assert_output_safe(
        output, primary_dir=primary_dir, profile_name=profile_name
    )
    primary_validation = (
        validate_primary_run(primary_dir) if require_primary else {"checks": {}}
    )
    primary_frame = (
        pd.read_csv(primary_dir / "lrt_raw.csv")
        if require_primary and (primary_dir / "lrt_raw.csv").exists()
        else None
    )
    primary_before = directory_fingerprints(primary_dir)

    # The publication path is claimed atomically so two full launches cannot
    # pass the earlier absence check and then overwrite each other's files.
    output.mkdir(parents=True, exist_ok=profile_name != "full")
    _write_run_config(output, profile_name, profile, jobs)
    write_source_manifest(output)
    (output / "source_provenance.md").write_text(
        "# Source provenance\n\n"
        "This is an independent fixed-R confirmation and does not replace or "
        "overwrite the primary 1,200-row LRT run.  The source manifest and "
        "seed-schedule audit were written before computation; the manifest "
        "also hashes the frozen analysis plan, tests, and write-back template. "
        "The independently frozen prelaunch manifest was verified before "
        "creating this directory.  The v1 LRT engine is imported unchanged; "
        "the hard-pinned loky process isolation is used only to collect "
        "additional scalar audit fields.\n",
        encoding="utf-8",
    )
    seed_audit = confirmation_seed_audit(profile.lrt_reps)
    (output / "seed_schedule_audit.json").write_text(
        json.dumps(seed_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    roots = confirmation_root_schedule(profile.lrt_reps)
    rows = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
        delayed(_confirmation_one)(profile, replicate, roots[replicate])
        for replicate in range(profile.lrt_reps)
    )
    frame = _write_rows(rows, output)
    summary = summarize_confirmation(
        frame, profile_name, primary_frame=primary_frame
    )
    summary.to_csv(output / f"{RAW_STEM}_summary.csv", index=False)
    (output / "writeback_fields.json").write_text(
        json.dumps(
            _json_clean(writeback_fields(summary.iloc[0])),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    primary_after = directory_fingerprints(primary_dir)
    primary_unchanged = primary_before == primary_after
    source_audit = validate_source_manifest(output)
    serialization_audit = validate_serializations(output)
    derived_output_audit = validate_derived_outputs(
        output, profile_name=profile_name, primary_frame=primary_frame
    )
    row_audit = validate_confirmation_rows(
        frame, profile_name=profile_name, profile=profile
    )
    audit = {
        "confirmation_protocol_version": CONFIRMATION_PROTOCOL_VERSION,
        "profile_name": profile_name,
        "engine": validate_locked_engine(
            require_pinned_hash=profile_name == "full"
        ),
        "seed_schedule": seed_audit,
        "source": source_audit,
        "frozen_prelaunch_source": frozen_source_audit,
        "serializations": serialization_audit,
        "derived_outputs": derived_output_audit,
        "rows": row_audit,
        "primary_validation": primary_validation,
        "primary_fingerprints_before": primary_before,
        "primary_fingerprints_after": primary_after,
        "primary_result_unchanged": primary_unchanged,
        "publication_ready": bool(
            profile_name == "full"
            and all(primary_validation["checks"].values())
            and primary_unchanged
            and source_audit["source_manifest_matches_current_code"]
            and frozen_source_audit["all_frozen_source_checks_passed"]
            and serialization_audit["row_counts_match"]
            and serialization_audit["required_scalar_values_match"]
            and derived_output_audit["summary_matches_recomputed_raw"]
            and derived_output_audit["writeback_matches_recomputed_summary"]
            and row_audit["all_structural_checks_passed"]
            and row_audit["checks"].get(
                "all_outer_analyses_valid_for_fixed_R_interpretation", False
            )
            and seed_audit["confirmation_direct_streams_unique"]
            and seed_audit["confirmation_direct_streams_disjoint_from_primary"]
        ),
    }
    (output / "confirmation_audit.json").write_text(
        json.dumps(_json_clean(audit), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    if not primary_unchanged:
        raise RuntimeError("primary LRT result changed during confirmation execution")
    if profile_name == "full" and not audit["publication_ready"]:
        raise RuntimeError("full confirmation failed its publication audit")
    return audit


def audit_existing_output(output: Path) -> dict[str, object]:
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    profile_name = config["profile_name"]
    profile = publication_profile() if profile_name == "full" else smoke_profile()
    frame = pd.read_csv(output / f"{RAW_STEM}_raw.csv")
    primary_frame = (
        pd.read_csv(PRIMARY_LRT_OUTPUT / "lrt_raw.csv")
        if (PRIMARY_LRT_OUTPUT / "lrt_raw.csv").exists()
        else None
    )
    summary = summarize_confirmation(
        frame, profile_name, primary_frame=primary_frame
    )
    return {
        "frozen_prelaunch_source": validate_frozen_source_manifest(),
        "source": validate_source_manifest(output),
        "serializations": validate_serializations(output),
        "derived_outputs": validate_derived_outputs(
            output, profile_name=profile_name, primary_frame=primary_frame
        ),
        "seed_schedule": confirmation_seed_audit(profile.lrt_reps),
        "rows": validate_confirmation_rows(
            frame, profile_name=profile_name, profile=profile
        ),
        "summary": summary.iloc[0].to_dict(),
        "writeback_fields": writeback_fields(summary.iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the immutable independent n=500 LRT null confirmation."
    )
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit an existing confirmation directory without running fits",
    )
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(_json_clean(audit_existing_output(args.output)), indent=2))
        return
    run_confirmation(
        profile_name=args.profile,
        output=args.output,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
