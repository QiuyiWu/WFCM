"""Importance-sampling and computational diagnostics for the WFCM revision.

The design answers the computational reviewer comment directly.  It separates
dimension scaling, sample-size scaling, and Monte Carlo-size sensitivity;
records proposal construction, importance drawing, and optimization times;
and reports ESS, maximum normalized weight, convergence, and center-coordinate
RMSE.  After label alignment, that metric is
``sqrt((k*d)^-1 * sum_{j,r} (estimated[j,r] - truth[j,r])^2)``.
Monte Carlo-size comparisons use independent proposal draws but identical
data and optimizer starts.  Thus their dispersion measures the Monte Carlo
variation that an actual rerun would incur, while avoiding a confounding
change in the local-optimization initialization.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from simulations.wfcm_core import (
    ISContext,
    WFCMParameterization,
    align_to_reference,
    average_nll_and_gradient,
    build_proposal,
    fit_wfcm,
    membership,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "revision_work" / "results" / "computation"


@dataclass(frozen=True)
class Profile:
    dimension_reps: int
    sample_size_reps: int
    mc_reps: int
    dimensions: tuple[int, ...]
    sample_sizes: tuple[int, ...]
    mc_sample_sizes: tuple[int, ...]
    mc_ratios: tuple[int, ...]
    n_dimension: int
    d_sample_size: int
    d_mc: int
    n_init: int
    maxiter: int


PROFILES = {
    "smoke": Profile(1, 1, 1, (2, 10), (150, 300), (150,), (1, 5), 150, 5, 5, 1, 120),
    "pilot": Profile(4, 4, 4, (2, 10, 25, 50), (250, 1_000), (250, 1_000), (1, 5, 20), 400, 10, 10, 2, 300),
    "full": Profile(20, 20, 20, (2, 10, 25, 50), (250, 1_000, 4_000), (250, 1_000), (1, 5, 20, 80), 500, 10, 10, 3, 450),
}


def _seed(seed: int, *indices: int) -> int:
    sequence = np.random.SeedSequence([int(seed), *(int(i) for i in indices)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def gaussian_mixture_sample(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-cluster imbalanced Gaussian design with mild anisotropy."""

    if d < 2:
        raise ValueError("dimension must be at least two")
    rng = np.random.default_rng(seed)
    weights = np.array([0.15, 0.30, 0.55])
    labels = rng.choice(3, size=n, p=weights)
    centers = np.zeros((3, d))
    centers[:, :2] = np.array([[-3.4, -1.8], [0.0, 3.0], [3.4, -1.5]])
    x = np.empty((n, d))
    for cluster in range(3):
        count = int(np.sum(labels == cluster))
        scales = np.full(d, 0.75)
        scales[:2] = (
            np.array([1.25, 0.60])
            if cluster == 0
            else np.array([0.65, 1.15])
            if cluster == 1
            else np.array([1.05, 0.70])
        )
        noise = rng.normal(size=(count, d)) * scales
        angle = (cluster - 1) * 0.45
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        noise[:, :2] = noise[:, :2] @ rotation.T
        x[labels == cluster] = centers[cluster] + noise
    return x, labels, centers


def _fit_record(
    x: np.ndarray,
    labels: np.ndarray,
    true_centers: np.ndarray,
    *,
    context: ISContext,
    proposal_seconds: float,
    draw_seconds: float,
    seed: int,
    n_init: int,
    maxiter: int,
) -> tuple[dict, object]:
    wall_start = perf_counter()
    fit = fit_wfcm(
        x,
        k=3,
        m=2.0,
        importance_size=context.size,
        proposal_kind="defensive",
        n_init=n_init,
        maxiter=maxiter,
        seed=seed,
        context=context,
    )
    wall_seconds = perf_counter() - wall_start
    aligned_centers, _, estimated_for_truth = align_to_reference(
        fit.params.centers, fit.params.weights, true_centers
    )
    estimated_labels = np.argmax(membership(x, fit.params.centers, fit.params.weights, 2.0), axis=1)
    # ARI is invariant to the estimated label names.  Center-coordinate RMSE
    # is computed after putting the centers in truth-label order; ``np.mean``
    # supplies the explicit 1/(k*d) normalization.
    center_rmse = float(np.sqrt(np.mean((aligned_centers - true_centers) ** 2)))
    record = {
        "n": x.shape[0],
        "d": x.shape[1],
        "importance_size": context.size,
        "proposal": context.proposal_name,
        "fit_success": fit.success,
        "optimizer_success": fit.optimizer_success,
        "gradient_inf_norm": fit.gradient_inf_norm,
        "iterations": fit.iterations,
        "function_evaluations": fit.function_evaluations,
        "proposal_seconds": proposal_seconds,
        "importance_draw_seconds": draw_seconds,
        "optimizer_seconds": fit.diagnostics.get(
            "optimizer_runtime_seconds", fit.runtime_seconds
        ),
        "fit_wall_seconds": wall_seconds,
        "total_seconds": proposal_seconds + draw_seconds + wall_seconds,
        "ess": fit.diagnostics["ess"],
        "ess_fraction": fit.diagnostics["ess_fraction"],
        "max_normalized_weight": fit.diagnostics["max_normalized_weight"],
        "average_nll_training_context": fit.average_nll,
        "center_rmse": center_rmse,
        "ari": float(adjusted_rand_score(labels, estimated_labels)),
        "sigma": fit.params.sigma,
        "weights_aligned": fit.params.weights[estimated_for_truth].tolist(),
        "centers_aligned": aligned_centers.tolist(),
    }
    return record, fit


def _single_context(
    x: np.ndarray,
    *,
    k: int,
    proposal_kind: str,
    importance_size: int,
    seed: int,
) -> tuple[ISContext, float, float]:
    start = perf_counter()
    proposal = build_proposal(x, k, proposal_kind, seed)
    proposal_seconds = perf_counter() - start
    start = perf_counter()
    context = ISContext.draw(proposal, importance_size, seed + 1)
    draw_seconds = perf_counter() - start
    return context, proposal_seconds, draw_seconds


def _dimension_one(profile: Profile, d: int, replicate: int, base_seed: int) -> list[dict]:
    n = profile.n_dimension
    data_seed = _seed(base_seed, 101, d, replicate)
    x, labels, centers = gaussian_mixture_sample(n, d, data_seed)
    rows = []
    for proposal_index, proposal_kind in enumerate(("defensive", "gmm_full", "gmm_diag")):
        m_is = 20 * n
        context, proposal_seconds, draw_seconds = _single_context(
            x,
            k=3,
            proposal_kind=proposal_kind,
            importance_size=m_is,
            seed=_seed(base_seed, 102, d, replicate, proposal_index),
        )
        record, _ = _fit_record(
            x,
            labels,
            centers,
            context=context,
            proposal_seconds=proposal_seconds,
            draw_seconds=draw_seconds,
            # Keep the local-optimization starts identical across proposals.
            seed=_seed(base_seed, 103, d, replicate),
            n_init=profile.n_init,
            maxiter=profile.maxiter,
        )
        record.update(
            {
                "experiment": "dimension_scaling",
                "replicate": replicate,
                "data_seed": data_seed,
                "proposal_kind": proposal_kind,
                "m_over_n": m_is / n,
            }
        )
        rows.append(record)
    return rows


def _sample_size_one(profile: Profile, n: int, replicate: int, base_seed: int) -> dict:
    d = profile.d_sample_size
    data_seed = _seed(base_seed, 111, n, replicate)
    x, labels, centers = gaussian_mixture_sample(n, d, data_seed)
    m_is = 20 * n
    context, proposal_seconds, draw_seconds = _single_context(
        x,
        k=3,
        proposal_kind="defensive",
        importance_size=m_is,
        seed=_seed(base_seed, 112, n, replicate),
    )
    record, _ = _fit_record(
        x,
        labels,
        centers,
        context=context,
        proposal_seconds=proposal_seconds,
        draw_seconds=draw_seconds,
        seed=_seed(base_seed, 113, n, replicate),
        n_init=profile.n_init,
        maxiter=profile.maxiter,
    )
    record.update(
        {
            "experiment": "sample_size_scaling",
            "replicate": replicate,
            "data_seed": data_seed,
            "proposal_kind": "defensive",
            "m_over_n": m_is / n,
        }
    )
    return record


def _mc_size_one(profile: Profile, n: int, replicate: int, base_seed: int) -> list[dict]:
    d = profile.d_mc
    data_seed = _seed(base_seed, 121, n, replicate)
    x, labels, centers = gaussian_mixture_sample(n, d, data_seed)
    proposal_start = perf_counter()
    proposal = build_proposal(x, 3, "defensive", _seed(base_seed, 122, n, replicate))
    proposal_seconds = perf_counter() - proposal_start

    # An independent, common scoring context avoids comparing the noisy
    # training criteria from different M values.
    score_draw_start = perf_counter()
    score_context = ISContext.draw(
        proposal,
        max(50_000, 40 * n),
        _seed(base_seed, 124, n, replicate),
    )
    score_draw_seconds = perf_counter() - score_draw_start
    fits = []
    rows = []
    common_fit_seed = _seed(base_seed, 125, n, replicate)
    for ratio in profile.mc_ratios:
        size = ratio * n
        draw_start = perf_counter()
        context = ISContext.draw(
            proposal,
            size,
            _seed(base_seed, 123, n, replicate, ratio),
        )
        draw_seconds = perf_counter() - draw_start
        record, fit = _fit_record(
            x,
            labels,
            centers,
            context=context,
            proposal_seconds=proposal_seconds,
            draw_seconds=draw_seconds,
            seed=common_fit_seed,
            n_init=profile.n_init,
            maxiter=profile.maxiter,
        )
        spec = WFCMParameterization(3, d, m_fixed=2.0)
        score_nll, _ = average_nll_and_gradient(spec.pack(fit.params), x, score_context, spec)
        record.update(
            {
                "experiment": "mc_size_sensitivity",
                "replicate": replicate,
                "data_seed": data_seed,
                "proposal_kind": "defensive",
                "m_over_n": ratio,
                "mc_draw_design": "independent_per_M",
                "optimizer_start_seed": common_fit_seed,
                "scoring_context_size": score_context.size,
                "scoring_context_draw_seconds_excluded_from_fit_runtime": score_draw_seconds,
                "average_nll_independent_scoring_context": float(score_nll),
            }
        )
        rows.append(record)
        fits.append((ratio, fit))

    successful_fits = [pair for pair in fits if pair[1].success]
    if not successful_fits:
        raise RuntimeError("all Monte Carlo-size fits failed convergence checks")
    reference_ratio, reference = max(successful_fits, key=lambda pair: pair[0])
    for row, (_, fit) in zip(rows, fits, strict=True):
        if not fit.success:
            row["reference_m_over_n"] = reference_ratio
            row["center_rmse_vs_largest_valid_m"] = np.nan
            row["weight_l1_vs_largest_valid_m"] = np.nan
            row["sigma_abs_error_vs_largest_valid_m"] = np.nan
            continue
        centers_aligned, weights_aligned, _ = align_to_reference(
            fit.params.centers, fit.params.weights, reference.params.centers
        )
        row["reference_m_over_n"] = reference_ratio
        # The same 1/(k*d) center-coordinate normalization is used for the
        # numerical-reference deviation; the raw field name is kept stable.
        row["center_rmse_vs_largest_valid_m"] = float(
            np.sqrt(np.mean((centers_aligned - reference.params.centers) ** 2))
        )
        row["weight_l1_vs_largest_valid_m"] = float(
            np.sum(np.abs(weights_aligned - reference.params.weights))
        )
        row["sigma_abs_error_vs_largest_valid_m"] = float(
            abs(fit.params.sigma - reference.params.sigma)
        )
    return rows


def _write(frame: pd.DataFrame, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{stem}_raw.csv", index=False)
    with (output / f"{stem}_raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in frame.to_dict(orient="records"):
            handle.write(json.dumps(_json_clean(row), sort_keys=True, allow_nan=False) + "\n")


def _json_clean(value):
    """Convert NumPy values and nonfinite floats to strict-JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summary(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    working = frame.assign(
        center_rmse_successful=frame["center_rmse"].where(frame["fit_success"]),
        ari_successful=frame["ari"].where(frame["fit_success"]),
    )
    return (
        working.groupby(group, as_index=False)
        .agg(
            median_total_seconds=("total_seconds", "median"),
            mean_total_seconds=("total_seconds", "mean"),
            median_proposal_seconds=("proposal_seconds", "median"),
            median_importance_draw_seconds=("importance_draw_seconds", "median"),
            median_optimizer_seconds=("optimizer_seconds", "median"),
            median_fit_wall_seconds=("fit_wall_seconds", "median"),
            mean_ess_fraction=("ess_fraction", "mean"),
            min_ess_fraction=("ess_fraction", "min"),
            mean_max_weight=("max_normalized_weight", "mean"),
            success_rate=("fit_success", "mean"),
            mean_center_rmse_all_endpoints=("center_rmse", "mean"),
            mean_center_rmse_successful=("center_rmse_successful", "mean"),
            mean_ari_all_endpoints=("ari", "mean"),
            mean_ari_successful=("ari_successful", "mean"),
            replicates=("replicate", "nunique"),
        )
    )


def run_dimension(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    tasks = [(d, r) for d in profile.dimensions for r in range(profile.dimension_reps)]
    chunks = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_dimension_one)(profile, d, r, seed) for d, r in tasks
    )
    rows = [row for chunk in chunks for row in chunk]
    frame = pd.DataFrame(rows)
    _write(frame, output, "dimension_scaling")
    _summary(frame, ["d", "proposal_kind"]).to_csv(
        output / "dimension_scaling_summary.csv", index=False
    )


def run_sample_size(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    tasks = [(n, r) for n in profile.sample_sizes for r in range(profile.sample_size_reps)]
    rows = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_sample_size_one)(profile, n, r, seed) for n, r in tasks
    )
    frame = pd.DataFrame(rows)
    _write(frame, output, "sample_size_scaling")
    _summary(frame, ["n"]).to_csv(output / "sample_size_scaling_summary.csv", index=False)


def run_mc_size(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    tasks = [(n, r) for n in profile.mc_sample_sizes for r in range(profile.mc_reps)]
    chunks = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_mc_size_one)(profile, n, r, seed) for n, r in tasks
    )
    rows = [row for chunk in chunks for row in chunk]
    frame = pd.DataFrame(rows)
    _write(frame, output, "mc_size_sensitivity")
    summary = _summary(frame, ["n", "m_over_n"])
    extra = (
        frame.assign(
            scoring_nll_successful=frame["average_nll_independent_scoring_context"].where(
                frame["fit_success"]
            )
        ).groupby(["n", "m_over_n"], as_index=False)
        .agg(
            mean_center_rmse_vs_largest_valid_m=(
                "center_rmse_vs_largest_valid_m", "mean"
            ),
            mean_weight_l1_vs_largest_valid_m=(
                "weight_l1_vs_largest_valid_m", "mean"
            ),
            mean_sigma_error_vs_largest_valid_m=(
                "sigma_abs_error_vs_largest_valid_m", "mean"
            ),
            mean_scoring_nll_all_endpoints=(
                "average_nll_independent_scoring_context", "mean"
            ),
            mean_scoring_nll_successful=("scoring_nll_successful", "mean"),
        )
    )
    summary.merge(extra, on=["n", "m_over_n"]).to_csv(
        output / "mc_size_sensitivity_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("dimension", "sample_size", "mc_size", "all"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    profile = PROFILES[args.profile]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "profile_name": args.profile,
                "profile": asdict(profile),
                "jobs": args.jobs,
                "seed": args.seed,
                "runtime_note": (
                    "perf_counter wall-clock seconds; jobs=1 is required for "
                    "uncontended runtime comparisons"
                ),
                "logical_cpu_count": os.cpu_count(),
                "machine": platform.machine(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if args.experiment in {"dimension", "all"}:
        run_dimension(profile, args.output, args.jobs, args.seed)
    if args.experiment in {"sample_size", "all"}:
        run_sample_size(profile, args.output, args.jobs, args.seed)
    if args.experiment in {"mc_size", "all"}:
        run_mc_size(profile, args.output, args.jobs, args.seed)


if __name__ == "__main__":
    main()
