"""Corrected rerun of the manuscript's retained asymptotic simulations.

This module preserves the three original questions--consistency, Gaussian
approximation, and grid selection of m--while repairing the prototype's
parameterization, label alignment, dependent MCMC data generation, repeated-
dataset bug, and self-whitening diagnostic.  Data are independent draws from a
high-resolution numerical inverse-CDF approximation to a one-dimensional WFCM
density.  Normality is standardized by an independent model-Fisher calculation
at the truth, never by the empirical covariance of the same Monte Carlo
estimates being assessed.
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

from simulations.wfcm_core import (
    FIT_PROJECTED_GRADIENT_TOLERANCE,
    ISContext,
    WFCMParameterization,
    WFCMParams,
    align_to_reference,
    build_proposal,
    energy_and_gradient_free,
    fit_wfcm,
    sample_wfcm_1d,
)
from simulations.task_checkpoint import (
    CHECKPOINT_PROTOCOL,
    run_checkpointed_tasks,
    write_or_validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "revision_work" / "results" / "legacy_corrected"
TRUTH = WFCMParams(
    sigma=0.9,
    centers=np.array([[-2.0], [2.0]]),
    weights=np.array([0.35, 0.65]),
    m=2.0,
)
M_GRID = (1.5, 1.7, 1.9, 2.0, 2.1, 2.3, 2.5)
PROPOSAL_KIND = "defensive"
LEGACY_PROTOCOL = "corrected-retained-asymptotic-simulations-v2-checkpointed"


@dataclass(frozen=True)
class Profile:
    estimation_reps: int
    m_selection_reps: int
    sample_sizes: tuple[int, ...]
    m_sample_sizes: tuple[int, ...]
    n_init: int
    maxiter: int
    truth_fisher_draws: int


PROFILES = {
    "smoke": Profile(3, 2, (100, 200), (100,), 1, 150, 10_000),
    "pilot": Profile(30, 20, (200, 500, 1_000), (200, 500), 2, 300, 100_000),
    "full": Profile(250, 200, (200, 500, 1_000, 2_000), (200, 500, 1_000), 3, 450, 750_000),
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
        "simulations/legacy_corrected_experiment.py",
        "simulations/task_checkpoint.py",
        "simulations/wfcm_core.py",
    )
    return {name: _file_sha256(ROOT / name) for name in relative}


def package_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-learn", "pandas", "joblib")
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def importance_size(n: int) -> int:
    """Use the floored M=n^(3/2) schedule, for which M/n tends to infinity."""

    return max(2_000, int(np.ceil(n ** 1.5)))


def _importance_cloud_sha256(context: ISContext) -> str:
    """Fingerprint the fixed samples and their proposal log densities."""

    digest = hashlib.sha256()
    digest.update(b"wfcm-importance-cloud-v1\0")
    for values in (context.samples, context.log_q):
        array = np.asarray(values, dtype="<f8", order="C")
        shape = np.asarray(array.shape, dtype="<i8")
        digest.update(np.asarray([array.ndim], dtype="<i8").tobytes())
        digest.update(shape.tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fit_failure_message(fit) -> str:
    """Name the numerical gate(s) that make a returned fit invalid."""

    failures: list[str] = []
    if not fit.optimizer_success:
        failures.append("optimizer_success is false")
    if not np.isfinite(fit.average_nll):
        failures.append("average_nll is nonfinite")
    free = np.asarray(fit.free_params, dtype=float)
    if np.any(~np.isfinite(free)):
        failures.append("free_params contain nonfinite values")
    if not np.isfinite(fit.gradient_inf_norm):
        failures.append("projected_gradient_inf_norm is nonfinite")
    elif fit.gradient_inf_norm > FIT_PROJECTED_GRADIENT_TOLERANCE:
        failures.append(
            "projected_gradient_inf_norm "
            f"{fit.gradient_inf_norm:.8g} exceeds "
            f"{FIT_PROJECTED_GRADIENT_TOLERANCE:.8g}"
        )
    if fit.success and not failures:
        return ""
    if not fit.success and not failures:
        failures.append("fit success flag is false for an unclassified numerical reason")
    optimizer_message = str(fit.message).strip()
    if optimizer_message:
        failures.append(f"optimizer message: {optimizer_message}")
    return "numerical gate failure: " + "; ".join(failures)


def truth_information(draws: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    spec = WFCMParameterization(k=2, d=1, m_fixed=TRUTH.m)
    z0 = spec.pack(TRUTH)
    x = sample_wfcm_1d(TRUTH, draws, seed, grid_size=200_001)
    _, gradients = energy_and_gradient_free(z0, x, spec)
    centered = gradients - np.mean(gradients, axis=0)
    information = centered.T @ centered / draws
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    floor = 1e-10 * max(float(np.max(eigenvalues)), 1.0)
    clipped = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    return information, inverse


def _load_or_create_truth_information(
    profile: Profile, output: Path, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Persist the common Fisher calculation atomically and reuse it on resume."""

    path = output / "truth_fisher.npz"
    expected_truth = WFCMParameterization(2, 1, m_fixed=TRUTH.m).pack(TRUTH)
    if path.exists():
        try:
            with np.load(path) as archive:
                if set(archive.files) != {
                    "information",
                    "inverse_information",
                    "truth_free",
                }:
                    raise RuntimeError("truth-Fisher archive has the wrong schema")
                information = np.asarray(archive["information"], dtype=float)
                inverse = np.asarray(archive["inverse_information"], dtype=float)
                truth_free = np.asarray(archive["truth_free"], dtype=float)
        except (OSError, ValueError) as exc:
            raise RuntimeError("existing truth-Fisher archive is unreadable") from exc
        if (
            information.shape != (4, 4)
            or inverse.shape != (4, 4)
            or truth_free.shape != (4,)
            or np.any(~np.isfinite(information))
            or np.any(~np.isfinite(inverse))
            or not np.array_equal(truth_free, expected_truth)
        ):
            raise RuntimeError("existing truth-Fisher archive failed validation")
        return information, inverse

    information, inverse = truth_information(
        profile.truth_fisher_draws, _seed(seed, 200)
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            information=information,
            inverse_information=inverse,
            truth_free=expected_truth,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return information, inverse


_PRACTICAL_GUARDRAIL_FIELDS = (
    "minimum_center_separation",
    "minimum_weight",
    "weight_floor",
    "weight_floor_margin",
    "fitted_log_sigma",
    "log_sigma_lower_bound",
    "log_sigma_upper_bound",
    "log_sigma_bound_margin",
    "minimum_free_coordinate_bound_margin",
    "minimum_relative_free_coordinate_bound_margin",
    "minimum_center_coordinate_bound_margin",
    "minimum_relative_center_coordinate_bound_margin",
    "minimum_weight_logit_bound_margin",
    "minimum_relative_weight_logit_bound_margin",
    "any_free_coordinate_at_or_near_bound",
    "theoretical_center_separation_constraint_enforced",
    "practical_box_bounds_are_data_adaptive",
)


def _missing_practical_guardrails() -> dict:
    values = {name: np.nan for name in _PRACTICAL_GUARDRAIL_FIELDS}
    values.update(
        {
            "any_free_coordinate_at_or_near_bound": False,
            "theoretical_center_separation_constraint_enforced": False,
            "practical_box_bounds_are_data_adaptive": True,
        }
    )
    return values


def _practical_guardrails(fit, x: np.ndarray) -> dict:
    """Describe observed interiority; these are diagnostics, not constraints."""

    params = fit.params
    spec = fit.parameterization
    if spec is None:
        spec = WFCMParameterization(
            params.k,
            params.d,
            m_fixed=params.m,
        )
    z = np.asarray(fit.free_params, dtype=float).reshape(-1)
    bounds = np.asarray(spec.bounds(x), dtype=float)
    if bounds.shape != (z.size, 2) or z.size != spec.size:
        raise ValueError("fit and practical-bound parameterization are incompatible")
    lower_distance = z - bounds[:, 0]
    upper_distance = bounds[:, 1] - z
    signed_margin = np.minimum(lower_distance, upper_distance)
    widths = bounds[:, 1] - bounds[:, 0]
    relative_margin = signed_margin / widths

    centers = np.asarray(params.centers, dtype=float)
    if params.k >= 2:
        center_distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        np.fill_diagonal(center_distances, np.inf)
        minimum_center_separation = float(np.min(center_distances))
    else:  # pragma: no cover - retained experiments use k=2
        minimum_center_separation = np.nan

    center_margin = signed_margin[spec.center_slice]
    center_relative_margin = relative_margin[spec.center_slice]
    logit_margin = signed_margin[spec.logit_slice]
    logit_relative_margin = relative_margin[spec.logit_slice]
    weight_floor_margin = float(np.min(params.weights) - spec.weight_floor)
    return {
        "minimum_center_separation": minimum_center_separation,
        "minimum_weight": float(np.min(params.weights)),
        "weight_floor": float(spec.weight_floor),
        "weight_floor_margin": weight_floor_margin,
        "fitted_log_sigma": float(z[0]),
        "log_sigma_lower_bound": float(bounds[0, 0]),
        "log_sigma_upper_bound": float(bounds[0, 1]),
        "log_sigma_bound_margin": float(signed_margin[0]),
        "minimum_free_coordinate_bound_margin": float(np.min(signed_margin)),
        "minimum_relative_free_coordinate_bound_margin": float(np.min(relative_margin)),
        "minimum_center_coordinate_bound_margin": float(np.min(center_margin)),
        "minimum_relative_center_coordinate_bound_margin": float(
            np.min(center_relative_margin)
        ),
        "minimum_weight_logit_bound_margin": (
            float(np.min(logit_margin)) if logit_margin.size else np.nan
        ),
        "minimum_relative_weight_logit_bound_margin": (
            float(np.min(logit_relative_margin)) if logit_relative_margin.size else np.nan
        ),
        "any_free_coordinate_at_or_near_bound": bool(
            np.min(relative_margin) <= 1e-6
        ),
        # The implemented optimizer uses data-adaptive box bounds and a weight
        # floor.  It does not impose the fixed separated-center stratum used in
        # the exact theorem; the observed separation is reported above.
        "theoretical_center_separation_constraint_enforced": False,
        "practical_box_bounds_are_data_adaptive": True,
    }


def _estimation_one(
    profile: Profile,
    n: int,
    replicate: int,
    base_seed: int,
    inverse_information: np.ndarray,
) -> list[dict]:
    data_seed = _seed(base_seed, 201, n, replicate)
    fit_seed = _seed(base_seed, 202, n, replicate)
    fit_rng = np.random.default_rng(fit_seed)
    proposal_seed = int(fit_rng.integers(0, 2**31 - 1))
    planned_importance_seed = int(fit_rng.integers(0, 2**31 - 1))
    importance_seed = planned_importance_seed
    m_is = importance_size(n)
    spec = WFCMParameterization(k=2, d=1, m_fixed=TRUTH.m)
    z0 = spec.pack(TRUTH)
    standard_errors = np.sqrt(np.diag(inverse_information) / n)
    parameter_names = (
        "log_sigma",
        "center_1",
        "center_2",
        "weight_floor_adjusted_log_ratio",
    )
    fit_start = perf_counter()
    try:
        x = sample_wfcm_1d(TRUTH, n, data_seed)
        fit = fit_wfcm(
            x,
            k=2,
            m=TRUTH.m,
            importance_size=m_is,
            proposal_kind=PROPOSAL_KIND,
            n_init=profile.n_init,
            maxiter=profile.maxiter,
            seed=fit_seed,
        )
        centers, weights, _ = align_to_reference(
            fit.params.centers, fit.params.weights, TRUTH.centers
        )
        aligned = WFCMParams(fit.params.sigma, centers, weights, TRUTH.m)
        zhat = spec.pack(aligned)
        difference = zhat - z0
        marginal_z = difference / standard_errors
        mahalanobis = float(
            n * difference @ np.linalg.pinv(inverse_information) @ difference
        )
        fit_success = bool(fit.success)
        optimizer_success = bool(fit.optimizer_success)
        gradient_inf_norm = float(fit.gradient_inf_norm)
        runtime_seconds = float(fit.runtime_seconds)
        ess = float(fit.diagnostics["ess"])
        ess_fraction = float(fit.diagnostics["ess_fraction"])
        max_normalized_weight = float(
            fit.diagnostics["max_normalized_weight"]
        )
        proposal_name = str(fit.proposal_name)
        importance_seed = int(fit.importance_seed)
        if importance_seed != planned_importance_seed:
            raise RuntimeError(
                "fit importance seed disagrees with the recorded seed schedule"
            )
        if fit.importance_context is None:
            raise RuntimeError("returned fit omitted its importance context")
        importance_cloud_sha256 = _importance_cloud_sha256(
            fit.importance_context
        )
        sigma_abs_error = abs(fit.params.sigma - TRUTH.sigma)
        center_rmse = float(np.sqrt(np.mean((centers - TRUTH.centers) ** 2)))
        weight_l1_error = float(np.sum(np.abs(weights - TRUTH.weights)))
        free_parameter_l2_error = float(np.linalg.norm(difference))
        practical_guardrails = _practical_guardrails(fit, x)
        failure_message = _fit_failure_message(fit)
        exception_type = ""
    except Exception as exc:
        # A single numerical failure is one attempted Monte Carlo replicate,
        # not a reason to discard every other scheduled replicate.  Preserve
        # its four-coordinate block so downstream denominators remain exact.
        zhat = np.full_like(z0, np.nan)
        marginal_z = np.full_like(z0, np.nan)
        mahalanobis = np.nan
        fit_success = False
        optimizer_success = False
        gradient_inf_norm = np.nan
        runtime_seconds = perf_counter() - fit_start
        ess = np.nan
        ess_fraction = np.nan
        max_normalized_weight = np.nan
        proposal_name = ""
        importance_cloud_sha256 = ""
        sigma_abs_error = np.nan
        center_rmse = np.nan
        weight_l1_error = np.nan
        free_parameter_l2_error = np.nan
        practical_guardrails = _missing_practical_guardrails()
        failure_message = f"{type(exc).__name__}: {exc}"
        exception_type = type(exc).__name__
    common = {
        "experiment": "estimation",
        "n": n,
        "replicate": replicate,
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "importance_size": m_is,
        "m_over_n": m_is / n,
        "fit_success": fit_success,
        "optimizer_success": optimizer_success,
        "gradient_inf_norm": gradient_inf_norm,
        "runtime_seconds": runtime_seconds,
        "ess": ess,
        "ess_fraction": ess_fraction,
        "max_normalized_weight": max_normalized_weight,
        "proposal_kind": PROPOSAL_KIND,
        "proposal_name": proposal_name,
        "proposal_seed": proposal_seed,
        "importance_seed": importance_seed,
        "importance_cloud_sha256": importance_cloud_sha256,
        "sigma_abs_error": sigma_abs_error,
        "center_rmse": center_rmse,
        "weight_l1_error": weight_l1_error,
        "free_parameter_l2_error": free_parameter_l2_error,
        "mahalanobis_squared": mahalanobis,
        "failure_message": failure_message,
        "exception_type": exception_type,
        **practical_guardrails,
    }
    return [
        {
            **common,
            "coordinate": name,
            "estimate_free": float(zhat[index]),
            "truth_free": float(z0[index]),
            "standard_error": float(standard_errors[index]),
            "marginal_z": float(marginal_z[index]),
        }
        for index, name in enumerate(parameter_names)
    ]


def _m_selection_one(profile: Profile, n: int, replicate: int, base_seed: int) -> list[dict]:
    data_seed = _seed(base_seed, 211, n, replicate)
    fit_seed = _seed(base_seed, 212, n, replicate)
    m_is = importance_size(n)
    # A single fixed IS cloud makes likelihood values directly comparable
    # over the m grid; maxiter is passed explicitly so the profile is honest.
    proposal_seed = _seed(fit_seed, 1)
    planned_importance_seed = _seed(fit_seed, 2)
    proposal_name = ""
    importance_seed = planned_importance_seed
    importance_cloud_sha256 = ""
    setup_start = perf_counter()
    try:
        x = sample_wfcm_1d(TRUTH, n, data_seed)
        proposal = build_proposal(x, 2, PROPOSAL_KIND, proposal_seed)
        context = ISContext.draw(proposal, m_is, planned_importance_seed)
        proposal_name = str(context.proposal_name)
        importance_seed = int(context.seed)
        if importance_seed != planned_importance_seed:
            raise RuntimeError(
                "common importance seed disagrees with the recorded seed schedule"
            )
        importance_cloud_sha256 = _importance_cloud_sha256(context)
        setup_failure = None
        setup_runtime = perf_counter() - setup_start
    except Exception as exc:
        context = None
        setup_failure = exc
        setup_runtime = perf_counter() - setup_start

    candidates: list[dict] = []
    for candidate in M_GRID:
        attempt_start = perf_counter()
        try:
            if setup_failure is not None:
                raise RuntimeError(
                    "common importance-cloud setup failed: "
                    f"{type(setup_failure).__name__}: {setup_failure}"
                )
            fit = fit_wfcm(
                x,
                k=2,
                m=candidate,
                importance_size=m_is,
                proposal_kind=PROPOSAL_KIND,
                n_init=profile.n_init,
                maxiter=profile.maxiter,
                # Common random starts isolate the change in m.
                seed=_seed(fit_seed, 3),
                context=context,
            )
            candidate_valid = bool(fit.success and np.isfinite(fit.average_nll))
            practical_guardrails = _practical_guardrails(fit, x)
            candidates.append(
                {
                    "candidate_m": candidate,
                    "candidate_fit_valid": candidate_valid,
                    "fit_success": bool(fit.success),
                    "optimizer_success": bool(fit.optimizer_success),
                    "average_nll": float(fit.average_nll),
                    "gradient_inf_norm": float(fit.gradient_inf_norm),
                    "runtime_seconds": float(fit.runtime_seconds),
                    "ess": float(fit.diagnostics["ess"]),
                    "ess_fraction": float(fit.diagnostics["ess_fraction"]),
                    "max_normalized_weight": float(
                        fit.diagnostics["max_normalized_weight"]
                    ),
                    "failure_message": (
                        "" if candidate_valid else _fit_failure_message(fit)
                    ),
                    "exception_type": "",
                    **practical_guardrails,
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "candidate_m": candidate,
                    "candidate_fit_valid": False,
                    "fit_success": False,
                    "optimizer_success": False,
                    "average_nll": np.nan,
                    "gradient_inf_norm": np.nan,
                    "runtime_seconds": perf_counter() - attempt_start,
                    "ess": np.nan,
                    "ess_fraction": np.nan,
                    "max_normalized_weight": np.nan,
                    "failure_message": f"{type(exc).__name__}: {exc}",
                    "exception_type": type(exc).__name__,
                    **_missing_practical_guardrails(),
                }
            )

    # Comparing a best-among-survivors subset changes the prespecified grid
    # rule.  Define the selection only when every candidate fit is valid.
    complete_grid_valid = all(item["candidate_fit_valid"] for item in candidates)
    if complete_grid_valid:
        best = min(candidates, key=lambda item: item["average_nll"])
        selected_m = float(best["candidate_m"])
        best_value = float(best["average_nll"])
    else:
        selected_m = np.nan
        best_value = np.nan

    return [
        {
            "experiment": "m_selection",
            "n": n,
            "replicate": replicate,
            "data_seed": data_seed,
            "fit_seed": fit_seed,
            "importance_size": m_is,
            "m_over_n": m_is / n,
            "proposal_kind": PROPOSAL_KIND,
            "proposal_name": proposal_name,
            "proposal_seed": proposal_seed,
            "importance_seed": importance_seed,
            "importance_cloud_sha256": importance_cloud_sha256,
            "true_m": TRUTH.m,
            "candidate_m": item["candidate_m"],
            "selected_m": selected_m,
            "selected_true_m": bool(
                complete_grid_valid and np.isclose(selected_m, TRUTH.m)
            ),
            "selection_defined": complete_grid_valid,
            "complete_grid_valid": complete_grid_valid,
            "candidate_fit_valid": item["candidate_fit_valid"],
            "eligible_for_selection": item["candidate_fit_valid"],
            "average_nll": item["average_nll"],
            "nll_above_best": (
                item["average_nll"] - best_value if complete_grid_valid else np.nan
            ),
            "fit_success": item["fit_success"],
            "optimizer_success": item["optimizer_success"],
            "gradient_inf_norm": item["gradient_inf_norm"],
            "runtime_seconds": item["runtime_seconds"],
            "ess": item["ess"],
            "ess_fraction": item["ess_fraction"],
            "max_normalized_weight": item["max_normalized_weight"],
            "common_cloud_setup_seconds": setup_runtime,
            "failure_message": item["failure_message"],
            "exception_type": item["exception_type"],
            **{name: item[name] for name in _PRACTICAL_GUARDRAIL_FIELDS},
        }
        for item in candidates
    ]


def _write_rows(rows: list[dict], output: Path, stem: str) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{stem}_raw.csv", index=False)
    with (output / f"{stem}_raw.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_clean(row), sort_keys=True, allow_nan=False) + "\n")
    return frame


def _json_clean(value):
    """Convert NumPy scalars and nonfinite values to strict JSON."""

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


def _validate_estimation_checkpoint_rows(task: tuple, rows: list[dict]) -> None:
    n, replicate = task
    expected_coordinates = [
        "log_sigma",
        "center_1",
        "center_2",
        "weight_floor_adjusted_log_ratio",
    ]
    if len(rows) != len(expected_coordinates):
        raise RuntimeError("legacy estimation checkpoint has the wrong row count")
    if [row.get("coordinate") for row in rows] != expected_coordinates:
        raise RuntimeError("legacy estimation checkpoint has the wrong coordinate order")
    if any(
        (row.get("n"), row.get("replicate")) != (n, replicate)
        or row.get("experiment") != "estimation"
        for row in rows
    ):
        raise RuntimeError("legacy estimation checkpoint has the wrong task identity")


def _validate_m_selection_checkpoint_rows(task: tuple, rows: list[dict]) -> None:
    n, replicate = task
    if len(rows) != len(M_GRID):
        raise RuntimeError("legacy m-selection checkpoint has the wrong row count")
    observed_grid = [float(row.get("candidate_m", np.nan)) for row in rows]
    if observed_grid != list(M_GRID):
        raise RuntimeError("legacy m-selection checkpoint has the wrong candidate order")
    if any(
        (row.get("n"), row.get("replicate")) != (n, replicate)
        or row.get("experiment") != "m_selection"
        for row in rows
    ):
        raise RuntimeError("legacy m-selection checkpoint has the wrong task identity")


def run_estimation(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _, inverse = _load_or_create_truth_information(profile, output, seed)
    tasks = [(n, r) for n in profile.sample_sizes for r in range(profile.estimation_reps)]
    if jobs == 1:
        chunks = run_checkpointed_tasks(
            tasks,
            output / "checkpoints" / "estimation",
            lambda task: _estimation_one(profile, *task, seed, inverse),
            _validate_estimation_checkpoint_rows,
        )
    else:
        chunks = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
            delayed(_estimation_one)(profile, n, r, seed, inverse) for n, r in tasks
        )
    rows = [row for chunk in chunks for row in chunk]
    frame = _write_rows(rows, output, "estimation")
    replicate_level = frame.drop_duplicates(["n", "replicate"])
    valid = replicate_level[replicate_level["fit_success"]].copy()
    summary = (
        valid.groupby("n", as_index=False)
        .agg(
            sigma_rmse=("sigma_abs_error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            mean_center_rmse=("center_rmse", "mean"),
            mean_weight_l1=("weight_l1_error", "mean"),
            mean_free_l2=("free_parameter_l2_error", "mean"),
            mean_mahalanobis_squared=("mahalanobis_squared", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_ess=("ess", "mean"),
            mean_ess_fraction=("ess_fraction", "mean"),
            mean_max_normalized_weight=("max_normalized_weight", "mean"),
            valid_replicates=("replicate", "nunique"),
        )
    )
    attempts = (
        replicate_level.groupby("n", as_index=False)
        .agg(
            success_rate=("fit_success", "mean"),
            attempted_replicates=("replicate", "nunique"),
        )
    )
    summary = attempts.merge(summary, on="n", how="left")
    for metric in ("sigma_rmse", "mean_center_rmse", "mean_weight_l1", "mean_free_l2"):
        usable = np.isfinite(summary[metric]) & (summary[metric] > 0.0)
        slope = (
            float(
                np.polyfit(
                    np.log(summary.loc[usable, "n"]),
                    np.log(summary.loc[usable, metric]),
                    1,
                )[0]
            )
            if int(np.sum(usable)) >= 2
            else np.nan
        )
        summary[f"loglog_slope_{metric}"] = slope
    summary.to_csv(output / "estimation_summary.csv", index=False)
    normality = (
        frame[frame["fit_success"]].groupby(["n", "coordinate"], as_index=False)
        .agg(
            mean_z=("marginal_z", "mean"),
            sd_z=("marginal_z", "std"),
            q025=("marginal_z", lambda x: float(np.quantile(x, 0.025))),
            q50=("marginal_z", "median"),
            q975=("marginal_z", lambda x: float(np.quantile(x, 0.975))),
            replicates=("replicate", "nunique"),
        )
    )
    normality.to_csv(output / "normality_summary.csv", index=False)


def run_m_selection(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    tasks = [(n, r) for n in profile.m_sample_sizes for r in range(profile.m_selection_reps)]
    if jobs == 1:
        chunks = run_checkpointed_tasks(
            tasks,
            output / "checkpoints" / "m_selection",
            lambda task: _m_selection_one(profile, *task, seed),
            _validate_m_selection_checkpoint_rows,
        )
    else:
        chunks = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
            delayed(_m_selection_one)(profile, n, r, seed) for n, r in tasks
        )
    rows = [row for chunk in chunks for row in chunk]
    frame = _write_rows(rows, output, "m_selection")
    selected = frame.drop_duplicates(["n", "replicate"])
    frequency = (
        selected.groupby(["n", "selected_m"], as_index=False)
        .agg(count=("replicate", "size"))
    )
    frequency["frequency"] = frequency["count"] / frequency.groupby("n")["count"].transform("sum")
    frequency.to_csv(output / "m_selection_frequency.csv", index=False)
    valid = frame[frame["eligible_for_selection"]]
    curves = (
        valid.groupby(["n", "candidate_m"], as_index=False)
        .agg(
            mean_nll_above_best=("nll_above_best", "mean"),
            se_nll_above_best=("nll_above_best", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
            mean_average_nll=("average_nll", "mean"),
            mean_ess=("ess", "mean"),
            mean_ess_fraction=("ess_fraction", "mean"),
            mean_max_normalized_weight=("max_normalized_weight", "mean"),
            valid_replicates=("replicate", "nunique"),
        )
    )
    attempts = (
        frame.groupby(["n", "candidate_m"], as_index=False)
        .agg(
            success_rate=("fit_success", "mean"),
            attempted_replicates=("replicate", "nunique"),
        )
    )
    curves = attempts.merge(curves, on=["n", "candidate_m"], how="left")
    curves.to_csv(output / "m_selection_curves.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("estimation", "m_selection", "all"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    if args.profile == "full" and args.jobs != 1:
        parser.error("the publication full profile requires --jobs 1 for runtime validity")
    args.output.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]
    config = {
        "legacy_protocol": LEGACY_PROTOCOL,
        "experiment": args.experiment,
        "profile_name": args.profile,
        "profile": asdict(profile),
        "truth": {
            "sigma": TRUTH.sigma,
            "centers": TRUTH.centers.tolist(),
            "weights": TRUTH.weights.tolist(),
            "m": TRUTH.m,
        },
        "m_grid": M_GRID,
        "proposal_kind": PROPOSAL_KIND,
        "fit_projected_gradient_tolerance": (
            FIT_PROJECTED_GRADIENT_TOLERANCE
        ),
        "importance_cloud_fingerprint": (
            "sha256(wfcm-importance-cloud-v1, little-endian float64 "
            "C-order samples and proposal log densities with shapes)"
        ),
        "jobs": args.jobs,
        "seed": args.seed,
        "checkpoint_protocol": (
            CHECKPOINT_PROTOCOL if args.jobs == 1 else "disabled when jobs > 1"
        ),
        "source_sha256": _source_sha256(),
        "package_versions": package_versions(),
        "runtime_note": (
            "perf_counter wall-clock seconds; jobs=1 is required for "
            "uncontended runtime summaries"
        ),
        "logical_cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    write_or_validate_config(args.output / "run_config.json", config)
    if args.experiment in {"estimation", "all"}:
        run_estimation(profile, args.output, args.jobs, args.seed)
    if args.experiment in {"m_selection", "all"}:
        run_m_selection(profile, args.output, args.jobs, args.seed)


if __name__ == "__main__":
    main()
