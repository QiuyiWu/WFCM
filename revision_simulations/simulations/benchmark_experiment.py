"""Factorial clustering benchmark requested by Reviewers 1 and 3.

The experiment crosses cluster geometry (spherical, anisotropic, and two
moons), imbalance (1:1, 4:1, and 10:1), and overlap (three levels).  It uses
the same generated dataset for WFCM and every comparator, and reports ARI,
optimal-label accuracy, minority recall, prototype error, discovered-cluster
count, noise fraction, failures, and runtime.  Center error for a method that
does not recover exactly k centers is reported as missing rather than being
made artificially favorable by dropping unmatched true centers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from simulations.competitors import DEFAULT_METHODS, fit_competitors
from simulations.data_generators import generate_scenario
from simulations.task_checkpoint import (
    CHECKPOINT_PROTOCOL,
    run_checkpointed_tasks,
    write_or_validate_config,
)
from simulations.wfcm_core import fit_wfcm, membership


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "revision_work" / "results" / "benchmark"
BENCHMARK_PROTOCOL = "reviewer-factorial-benchmark-v2-checkpointed"


@dataclass(frozen=True)
class Profile:
    replicates: int
    n: int
    competitor_n_init: int
    wfcm_n_init: int
    wfcm_importance_size: int
    maxiter: int
    scenarios: tuple[str, ...] = ("spherical", "anisotropic", "irregular")
    imbalances: tuple[str, ...] = ("balanced", "moderate", "severe")
    overlaps: tuple[str, ...] = ("low", "moderate", "high")


PROFILES = {
    "smoke": Profile(1, 180, 2, 1, 3_000, 120, ("spherical", "irregular"), ("balanced",), ("moderate",)),
    "pilot": Profile(5, 400, 5, 2, 6_000, 300),
    # With 100 replicates, the worst-case Monte Carlo SE of a cellwise
    # proportion is five percentage points; continuous metrics are paired
    # across methods, which materially improves comparison precision.
    # Use the same number of randomized starts for WFCM and every comparator
    # that exposes a restart control.  This avoids confounding clustering
    # quality or runtime with an asymmetric multistart budget.
    "full": Profile(100, 600, 10, 10, 12_000, 450),
}


def _seed(seed: int, *indices: int) -> int:
    sequence = np.random.SeedSequence([int(seed), *(int(i) for i in indices)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    relative = (
        "simulations/benchmark_experiment.py",
        "simulations/competitors.py",
        "simulations/data_generators.py",
        "simulations/task_checkpoint.py",
        "simulations/wfcm_core.py",
    )
    return {name: _file_sha256(ROOT / name) for name in relative}


def _competitor_options(profile: Profile) -> dict[str, dict]:
    return {
        "fcm": {
            "n_init": profile.competitor_n_init,
            "max_iter": profile.maxiter,
            "m": 2.0,
        },
        "polynomial_fcm": {
            "n_init": profile.competitor_n_init,
            "max_iter": profile.maxiter,
            "beta": 0.5,
        },
        "kmeans": {
            "n_init": profile.competitor_n_init,
            "max_iter": profile.maxiter,
        },
        "ekmeans": {
            "n_init": profile.competitor_n_init,
            "max_iter": profile.maxiter,
        },
        "spectral": {
            "n_neighbors": 15,
            "n_init": profile.competitor_n_init,
        },
        "gmm_full": {
            "n_init": profile.competitor_n_init,
            "max_iter": profile.maxiter,
        },
    }


def _effective_method_metadata(profile: Profile) -> dict[str, dict]:
    """Archive effective controls, including defaults not passed explicitly."""

    return {
        "wfcm": {
            "m": 2.0,
            "proposal_kind": "defensive",
            "importance_size": profile.wfcm_importance_size,
            "n_init": profile.wfcm_n_init,
            "maxiter": profile.maxiter,
        },
        "fcm": {**_competitor_options(profile)["fcm"], "tol": 1e-6},
        "kmeans": {
            **_competitor_options(profile)["kmeans"],
            "init": "k-means++",
            "tol": 1e-4,
            "algorithm": "lloyd",
        },
        "ekmeans": {
            **_competitor_options(profile)["ekmeans"],
            "alpha": "dvariance",
            "scale": 2.0,
            "use_numba": False,
        },
        "polynomial_fcm": {
            **_competitor_options(profile)["polynomial_fcm"],
            "tol": 1e-6,
        },
        "hdbscan": {
            "min_cluster_size": max(5, int(np.ceil(0.02 * profile.n))),
            "effective_min_samples": max(5, int(np.ceil(0.02 * profile.n))),
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
        },
        "agglomerative": {"linkage": "ward"},
        "spectral": {
            **_competitor_options(profile)["spectral"],
            "affinity": "nearest_neighbors",
            "assign_labels": "kmeans",
            "eigen_solver": "arpack",
        },
        "gmm_full": {
            **_competitor_options(profile)["gmm_full"],
            "covariance_type": "full",
            "init_params": "kmeans",
            "reg_covar": "max(1e-8, 1e-6 * (1 + mean marginal variance))",
        },
    }


def _label_metrics(
    truth: np.ndarray, predicted: np.ndarray, k: int
) -> tuple[float, float, float, dict[int, int]]:
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    observed = np.unique(predicted[predicted >= 0])
    if observed.size == 0:
        return 0.0, 0.0, 0.0, {}
    confusion = np.zeros((k, observed.size), dtype=np.int64)
    for true_label in range(k):
        for column, predicted_label in enumerate(observed):
            confusion[true_label, column] = int(
                np.sum((truth == true_label) & (predicted == predicted_label))
            )
    rows, columns = linear_sum_assignment(-confusion)
    mapping = {int(observed[column]): int(row) for row, column in zip(rows, columns, strict=True)}
    correct = int(sum(confusion[row, column] for row, column in zip(rows, columns, strict=True)))
    recalls = np.zeros(k, dtype=float)
    for row, column in zip(rows, columns, strict=True):
        denominator = int(np.sum(truth == row))
        recalls[row] = confusion[row, column] / denominator
    counts = np.bincount(truth, minlength=k)
    minority = counts == counts.min()
    return correct / truth.size, float(np.mean(recalls)), float(np.mean(recalls[minority])), mapping


def _center_metrics(
    estimated: np.ndarray, truth: np.ndarray
) -> tuple[float, float, int, bool]:
    estimated = np.asarray(estimated, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimated.ndim != 2 or estimated.shape[1:] != truth.shape[1:] or estimated.shape[0] == 0:
        return float("nan"), float("nan"), 0, False
    cost = np.linalg.norm(estimated[:, None, :] - truth[None, :, :], axis=2)
    estimated_rows, truth_columns = linear_sum_assignment(cost)
    matched = len(estimated_rows)
    # Prototype RMSE is sqrt{matched^{-1} sum_j ||vhat_j-v_j||^2}.  Taking
    # the mean over both centers and coordinates would introduce an unintended
    # extra factor 1/sqrt(d).
    matched_squared_distances = np.sum(
        (estimated[estimated_rows] - truth[truth_columns]) ** 2,
        axis=1,
    )
    matched_rmse = float(np.sqrt(np.mean(matched_squared_distances)))
    exact_count = estimated.shape[0] == truth.shape[0]
    exact_rmse = matched_rmse if exact_count else float("nan")
    return exact_rmse, matched_rmse, matched, exact_count


def _metric_record(
    *,
    method: str,
    labels: np.ndarray,
    centers: np.ndarray,
    runtime: float,
    success: bool,
    message: str,
    noise_fraction: float,
    scenario,
    diagnostics: dict,
) -> dict:
    accuracy, macro_recall, minority_recall, _ = _label_metrics(
        scenario.labels, labels, scenario.k
    )
    center_rmse, matched_center_rmse, matched_centers, exact_center_count = _center_metrics(
        centers, scenario.true_centers
    )
    return {
        "method": method,
        "success": bool(success),
        "message": message,
        "runtime_seconds": float(runtime),
        "ari": float(adjusted_rand_score(scenario.labels, labels)),
        "optimal_label_accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "minority_recall": float(minority_recall),
        "center_rmse_exact_k": center_rmse,
        "matched_center_rmse": matched_center_rmse,
        "matched_centers": matched_centers,
        "exact_center_count": exact_center_count,
        "n_clusters_found": int(np.unique(labels[labels >= 0]).size),
        "noise_fraction": float(noise_fraction),
        "diagnostics": diagnostics,
    }


def _one_dataset(
    profile: Profile,
    scenario_name: str,
    imbalance: str,
    overlap: str,
    replicate: int,
    base_seed: int,
) -> list[dict]:
    scenario_index = profile.scenarios.index(scenario_name)
    imbalance_index = profile.imbalances.index(imbalance)
    overlap_index = profile.overlaps.index(overlap)
    data_seed = _seed(
        base_seed, 301, scenario_index, imbalance_index, overlap_index, replicate
    )
    scenario = generate_scenario(
        scenario_name,
        n=profile.n,
        k=2,
        d=2,
        imbalance=imbalance,
        overlap=overlap,
        seed=data_seed,
    )
    common = {
        "experiment": "clustering_benchmark",
        "scenario": scenario.scenario,
        "imbalance": imbalance,
        "imbalance_ratio_target": scenario.imbalance_ratio,
        "imbalance_ratio_realized": scenario.metadata["realized_majority_to_minority_ratio"],
        "overlap": overlap,
        "overlap_value": (
            scenario.metadata["noise_sd"]
            if scenario.scenario == "irregular"
            else scenario.metadata["separation"]
        ),
        "n": scenario.n,
        "d": scenario.d,
        "k": scenario.k,
        "replicate": replicate,
        "data_seed": data_seed,
        "cluster_counts": scenario.counts.tolist(),
    }
    rows = []

    wfcm_seed = _seed(base_seed, 302, scenario_index, imbalance_index, overlap_index, replicate)
    start = perf_counter()
    try:
        wfcm = fit_wfcm(
            scenario.x,
            k=scenario.k,
            m=2.0,
            importance_size=profile.wfcm_importance_size,
            proposal_kind="defensive",
            n_init=profile.wfcm_n_init,
            maxiter=profile.maxiter,
            seed=wfcm_seed,
        )
        wfcm_labels = np.argmax(
            membership(scenario.x, wfcm.params.centers, wfcm.params.weights, wfcm.params.m),
            axis=1,
        ).astype(np.int64)
        wfcm_record = _metric_record(
            method="wfcm",
            labels=wfcm_labels,
            centers=wfcm.params.centers,
            runtime=perf_counter() - start,
            success=wfcm.success,
            message=wfcm.message,
            noise_fraction=0.0,
            scenario=scenario,
            diagnostics={
                "sigma": wfcm.params.sigma,
                "weights": wfcm.params.weights.tolist(),
                "m": wfcm.params.m,
                "average_nll": wfcm.average_nll,
                "gradient_inf_norm": wfcm.gradient_inf_norm,
                "optimizer_success": wfcm.optimizer_success,
                "ess": wfcm.diagnostics["ess"],
                "ess_fraction": wfcm.diagnostics["ess_fraction"],
                "max_normalized_weight": wfcm.diagnostics["max_normalized_weight"],
                "importance_size": profile.wfcm_importance_size,
                "proposal": wfcm.proposal_name,
            },
        )
    except Exception as exc:
        wfcm_record = _metric_record(
            method="wfcm",
            labels=np.full(scenario.n, -1, dtype=np.int64),
            centers=np.empty((0, scenario.d)),
            runtime=perf_counter() - start,
            success=False,
            message=f"{type(exc).__name__}: {exc}",
            noise_fraction=1.0,
            scenario=scenario,
            diagnostics={"exception_type": type(exc).__name__},
        )
    rows.append({**common, "method_seed": wfcm_seed, **wfcm_record})

    competitor_seed = _seed(
        base_seed, 303, scenario_index, imbalance_index, overlap_index, replicate
    )
    options = _competitor_options(profile)
    results = fit_competitors(
        scenario.x,
        scenario.k,
        methods=DEFAULT_METHODS,
        seed=competitor_seed,
        method_options=options,
    )
    for method in DEFAULT_METHODS:
        result = results[method]
        rows.append(
            {
                **common,
                "method_seed": result.seed,
                **_metric_record(
                    method=method,
                    labels=result.labels,
                    centers=result.centers,
                    runtime=result.runtime_seconds,
                    success=result.success,
                    message=result.message,
                    noise_fraction=result.noise_fraction,
                    scenario=scenario,
                    diagnostics=result.diagnostics,
                ),
            }
        )
    return rows


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_clean(value):
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


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Write an explicitly denominator-aware convenience summary.

    Exception records deliberately contain all-noise placeholder assignments so
    that the raw file retains one row per scheduled method/data-set pair.  Those
    placeholders are not scientific endpoint estimates.  Consequently every
    clustering-quality mean below is restricted to method-successful, finite
    endpoints, while success and runtime retain every attempted fit.

    Manuscript-facing summaries are independently recomputed and more fully
    audited by ``revision_work/generated/benchmark/make_benchmark_outputs.py``.
    """

    def mean_se(values: pd.Series) -> tuple[float, float, int]:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        numeric = numeric[np.isfinite(numeric)]
        denominator = int(numeric.size)
        if denominator == 0:
            return np.nan, np.nan, 0
        mean = float(np.mean(numeric))
        se = (
            float(np.std(numeric, ddof=1) / np.sqrt(denominator))
            if denominator > 1
            else np.nan
        )
        return mean, se, denominator

    grouping = ["scenario", "imbalance", "overlap", "method"]
    rows: list[dict] = []
    metrics = {
        "ari": "ari",
        "accuracy": "optimal_label_accuracy",
        "macro_recall": "macro_recall",
        "minority_recall": "minority_recall",
        "matched_center_rmse": "matched_center_rmse",
        "noise_fraction": "noise_fraction",
        "clusters_found": "n_clusters_found",
    }
    for keys, group in frame.groupby(grouping, sort=False):
        success = group["success"].fillna(False).astype(bool)
        attempted = int(len(group))
        successful = int(success.sum())
        success_rate = successful / attempted if attempted else np.nan
        row = {
            **dict(zip(grouping, keys, strict=True)),
            "attempted_replicates": attempted,
            "successful_replicates": successful,
            "failed_replicates": attempted - successful,
            "success_rate": success_rate,
            "success_rate_mcse": (
                float(np.sqrt(success_rate * (1.0 - success_rate) / attempted))
                if attempted else np.nan
            ),
            "metric_endpoint_policy": "method-successful finite endpoints only",
        }
        for label, column in metrics.items():
            mean, se, denominator = mean_se(group.loc[success, column])
            row[f"mean_{label}_successful"] = mean
            row[f"se_{label}_successful"] = se
            row[f"{label}_successful_defined_replicates"] = denominator

        exact_mask = success & group["exact_center_count"].fillna(False).astype(bool)
        exact_mean, exact_se, exact_denominator = mean_se(
            group.loc[exact_mask, "center_rmse_exact_k"]
        )
        row.update(
            {
                "mean_center_rmse_exact_k_successful": exact_mean,
                "se_center_rmse_exact_k_successful": exact_se,
                "center_rmse_exact_k_successful_defined_replicates": exact_denominator,
            }
        )
        if successful:
            exact_count = int(
                group.loc[success, "exact_center_count"].fillna(False).astype(bool).sum()
            )
            exact_rate = exact_count / successful
            exact_rate_mcse = float(
                np.sqrt(exact_rate * (1.0 - exact_rate) / successful)
            )
        else:
            exact_count = 0
            exact_rate = np.nan
            exact_rate_mcse = np.nan
        runtime_mean, runtime_se, runtime_denominator = mean_se(group["runtime_seconds"])
        row.update(
            {
                "exact_center_count_successful_replicates": exact_count,
                "exact_center_count_successful_denominator": successful,
                "exact_center_count_rate_among_successful": exact_rate,
                "exact_center_count_rate_mcse": exact_rate_mcse,
                "mean_runtime_seconds_all_attempts": runtime_mean,
                "se_runtime_seconds_all_attempts": runtime_se,
                "runtime_defined_attempts": runtime_denominator,
                "median_runtime_seconds_all_attempts": float(
                    np.nanmedian(pd.to_numeric(group["runtime_seconds"], errors="coerce"))
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def package_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "scipy",
        "scikit-learn",
        "sklekmeans",
        "pandas",
        "joblib",
    )
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _validate_checkpoint_rows(task: tuple, rows: list[dict]) -> None:
    scenario, imbalance, overlap, replicate = task
    expected_methods = ("wfcm", *DEFAULT_METHODS)
    if len(rows) != len(expected_methods):
        raise RuntimeError("benchmark checkpoint has the wrong number of methods")
    observed_methods = [row.get("method") for row in rows]
    if observed_methods != list(expected_methods):
        raise RuntimeError("benchmark checkpoint has the wrong method order")
    for row in rows:
        observed = (
            row.get("scenario"),
            row.get("imbalance"),
            row.get("overlap"),
            row.get("replicate"),
        )
        if observed != (scenario, imbalance, overlap, replicate):
            raise RuntimeError("benchmark checkpoint row has the wrong task identity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    if args.profile == "full" and args.jobs != 1:
        parser.error("the publication full profile requires --jobs 1 for runtime validity")
    profile = PROFILES[args.profile]
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "profile_name": args.profile,
        "profile": asdict(profile),
        "methods": ("wfcm", *DEFAULT_METHODS),
        "method_execution_order": ("wfcm", *DEFAULT_METHODS),
        "method_options_effective": _effective_method_metadata(profile),
        "jobs": args.jobs,
        "seed": args.seed,
        "checkpoint_protocol": (
            CHECKPOINT_PROTOCOL if args.jobs == 1 else "disabled when jobs > 1"
        ),
        "source_sha256": _source_sha256(),
        "package_versions": package_versions(),
        "runtime_note": (
            "perf_counter wall-clock seconds; jobs=1 is required for "
            "uncontended cross-method runtime comparisons"
        ),
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    write_or_validate_config(args.output / "run_config.json", config)
    tasks = [
        (scenario, imbalance, overlap, replicate)
        for scenario in profile.scenarios
        for imbalance in profile.imbalances
        for overlap in profile.overlaps
        for replicate in range(profile.replicates)
    ]
    if args.jobs == 1:
        chunks = run_checkpointed_tasks(
            tasks,
            args.output / "checkpoints" / "benchmark",
            lambda task: _one_dataset(profile, *task, args.seed),
            _validate_checkpoint_rows,
        )
    else:
        chunks = Parallel(n_jobs=args.jobs, prefer="threads", verbose=10)(
            delayed(_one_dataset)(
                profile, scenario, imbalance, overlap, replicate, args.seed
            )
            for scenario, imbalance, overlap, replicate in tasks
        )
    rows = [row for chunk in chunks for row in chunk]
    with (args.output / "benchmark_raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_clean(row), sort_keys=True, allow_nan=False, default=_json_default)
                + "\n"
            )
    csv_rows = [{key: value for key, value in row.items() if key != "diagnostics"} for row in rows]
    frame = pd.DataFrame(csv_rows)
    frame.to_csv(args.output / "benchmark_raw.csv", index=False)
    summarize(frame).to_csv(args.output / "benchmark_summary.csv", index=False)


if __name__ == "__main__":
    main()
