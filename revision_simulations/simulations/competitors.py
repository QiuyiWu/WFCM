"""Competitor implementations for the reviewer-requested clustering benchmark.

Every method is exposed through :func:`fit_competitor` and returns the same
``ClusteringResult`` record.  Algorithm failures are recorded rather than
silently dropped, which lets the benchmark report failure rates.  The
polynomial-fuzzifier implementation follows Klawonn--Hoppner/Winkler et al.:

    f_beta(u) = (1-beta)/(1+beta) u^2 + 2 beta/(1+beta) u,

with the active-set membership update required to enforce nonnegative fuzzy
memberships.  It is distinct from possibilistic c-means, which is sometimes
also abbreviated PFCM.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import softmax
from sklearn.cluster import (
    AgglomerativeClustering,
    HDBSCAN,
    KMeans,
    SpectralClustering,
    kmeans_plusplus,
)
from sklearn.mixture import GaussianMixture


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
POLYNOMIAL_BETA_MAX = 1.0 - 1e-10

DEFAULT_METHODS = (
    "fcm",
    "kmeans",
    "ekmeans",
    "polynomial_fcm",
    "hdbscan",
    "agglomerative",
    "spectral",
    "gmm_full",
)


def _canonical_method(method: str) -> str:
    canonical = method.strip().lower().replace("-", "_")
    aliases = {
        "standard_fcm": "fcm",
        "sklearn_kmeans": "kmeans",
        "equilibrium_kmeans": "ekmeans",
        "poly_fcm": "polynomial_fcm",
        "hierarchical": "agglomerative",
        "full_gmm": "gmm_full",
    }
    return aliases.get(canonical, canonical)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _as_data(x: ArrayLike) -> FloatArray:
    data = np.asarray(x, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("data must be a nonempty n-by-d array")
    if np.any(~np.isfinite(data)):
        raise ValueError("data contain nonfinite values")
    return data


def _validate_k(k: int, n: int) -> int:
    if not isinstance(k, (int, np.integer)) or k < 2 or k > n:
        raise ValueError("k must be an integer between two and n")
    return int(k)


def _squared_distances(data: FloatArray, centers: FloatArray) -> FloatArray:
    differences = data[:, None, :] - centers[None, :, :]
    return np.einsum("nkd,nkd->nk", differences, differences, optimize=True)


def _memberships_from_inverse_power(d2: FloatArray, exponent: float) -> FloatArray:
    """Return normalized ``d2**(-exponent)`` with exact-center handling."""

    memberships = np.zeros_like(d2)
    zero_rows = np.any(d2 == 0.0, axis=1)
    for row in np.flatnonzero(zero_rows):
        at_center = d2[row] == 0.0
        memberships[row, at_center] = 1.0 / int(np.sum(at_center))
    ordinary = ~zero_rows
    if np.any(ordinary):
        with np.errstate(divide="ignore"):
            log_terms = -exponent * np.log(d2[ordinary])
        memberships[ordinary] = softmax(log_terms, axis=1)
    return memberships


def fcm_membership(x: ArrayLike, centers: ArrayLike, m: float = 2.0) -> FloatArray:
    """Standard FCM memberships for squared Euclidean distances."""

    data = _as_data(x)
    prototypes = np.asarray(centers, dtype=float)
    if prototypes.ndim == 1:
        prototypes = prototypes[:, None]
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] < 2
        or prototypes.shape[1] != data.shape[1]
        or np.any(~np.isfinite(prototypes))
    ):
        raise ValueError("centers and data have incompatible shapes")
    if not np.isfinite(m) or m <= 1.0:
        raise ValueError("m must be finite and larger than one")
    return _memberships_from_inverse_power(_squared_distances(data, prototypes), 1.0 / (m - 1.0))


def polynomial_fuzzifier(u: ArrayLike, beta: float) -> FloatArray:
    """Evaluate the quadratic polynomial fuzzifier ``f_beta``."""

    values = np.asarray(u, dtype=float)
    if not np.isfinite(beta) or not 0.0 <= beta <= POLYNOMIAL_BETA_MAX:
        raise ValueError("beta must lie in [0, 1 - 1e-10]")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("memberships must lie in [0,1]")
    return ((1.0 - beta) * values**2 + 2.0 * beta * values) / (1.0 + beta)


def polynomial_membership(x: ArrayLike, centers: ArrayLike, beta: float = 0.5) -> FloatArray:
    """Membership update for FCM with the quadratic polynomial fuzzifier.

    For each observation, centers are sorted by distance and the largest
    active set yielding positive memberships is retained, as in equation (3)
    of Winkler, Klawonn, and Kruse (2011).  ``beta=0`` reduces exactly to FCM
    with ``m=2``; as ``beta`` approaches one, assignments become crisper.
    """

    data = _as_data(x)
    prototypes = np.asarray(centers, dtype=float)
    if prototypes.ndim == 1:
        prototypes = prototypes[:, None]
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] < 2
        or prototypes.shape[1] != data.shape[1]
        or np.any(~np.isfinite(prototypes))
    ):
        raise ValueError("centers and data have incompatible shapes")
    if not np.isfinite(beta) or not 0.0 <= beta <= POLYNOMIAL_BETA_MAX:
        raise ValueError("beta must lie in [0, 1 - 1e-10]")

    d2 = _squared_distances(data, prototypes)
    memberships = np.zeros_like(d2)
    zero_rows = np.any(d2 == 0.0, axis=1)
    for row in np.flatnonzero(zero_rows):
        at_center = d2[row] == 0.0
        memberships[row, at_center] = 1.0 / int(np.sum(at_center))

    denominator = 1.0 - beta
    for row in np.flatnonzero(~zero_rows):
        order = np.argsort(d2[row], kind="stable")
        sorted_d2 = d2[row, order]
        inverse = 1.0 / sorted_d2
        chosen: FloatArray | None = None
        chosen_size = 0
        # Starting with all centers and removing the farthest nonpositive
        # one implements the ordered active-set rule in the paper.
        for active_size in range(prototypes.shape[0], 0, -1):
            normalizer = (1.0 + (active_size - 1) * beta) / np.sum(inverse[:active_size])
            candidate = (normalizer * inverse[:active_size] - beta) / denominator
            if active_size == 1 or candidate[-1] > 0.0:
                chosen = np.maximum(candidate, 0.0)
                chosen_size = active_size
                break
        if chosen is None:  # pragma: no cover - active_size=1 is always feasible
            raise FloatingPointError("polynomial membership active-set update failed")
        chosen /= chosen.sum()
        memberships[row, order[:chosen_size]] = chosen

    if np.any(~np.isfinite(memberships)) or not np.allclose(
        memberships.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12
    ):
        raise FloatingPointError("polynomial membership calculation failed")
    return memberships


@dataclass
class ClusteringResult:
    method: str
    labels: IntArray
    centers: FloatArray
    memberships: FloatArray | None
    runtime_seconds: float
    success: bool
    message: str
    requested_k: int
    n_clusters_found: int
    noise_fraction: float
    seed: int
    diagnostics: dict[str, Any]

    def to_record(self, include_assignments: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "method": self.method,
            "runtime_seconds": self.runtime_seconds,
            "success": self.success,
            "message": self.message,
            "requested_k": self.requested_k,
            "n_clusters_found": self.n_clusters_found,
            "noise_fraction": _json_ready(self.noise_fraction),
            "seed": self.seed,
            "centers": self.centers.tolist(),
            "diagnostics": _json_ready(self.diagnostics),
        }
        if include_assignments:
            record["labels"] = self.labels.tolist()
            record["memberships"] = None if self.memberships is None else self.memberships.tolist()
        return record


@dataclass
class _FuzzyCandidate:
    centers: FloatArray
    memberships: FloatArray
    objective: float
    iterations: int
    converged: bool
    objective_history: list[float]


def _initial_centers(data: FloatArray, k: int, seed: int) -> FloatArray:
    if np.unique(data, axis=0).shape[0] < k:
        raise ValueError("fuzzy clustering requires at least k distinct observations")
    centers, _ = kmeans_plusplus(data, n_clusters=k, random_state=seed)
    return np.asarray(centers, dtype=float)


def _fit_fuzzy_candidate(
    data: FloatArray,
    k: int,
    seed: int,
    max_iter: int,
    tol: float,
    method: str,
    m: float,
    beta: float,
) -> _FuzzyCandidate:
    centers = _initial_centers(data, k, seed)
    history: list[float] = []
    converged = False
    memberships = np.empty((data.shape[0], k), dtype=float)

    for iteration in range(1, max_iter + 1):
        if method == "fcm":
            memberships = fcm_membership(data, centers, m)
            prototype_weights = memberships**m
        else:
            memberships = polynomial_membership(data, centers, beta)
            prototype_weights = polynomial_fuzzifier(memberships, beta)
        totals = prototype_weights.sum(axis=0)
        if np.any(totals <= np.finfo(float).tiny):
            raise FloatingPointError("a fuzzy cluster received zero total weight")
        updated = (prototype_weights.T @ data) / totals[:, None]
        if method == "fcm":
            updated_memberships = fcm_membership(data, updated, m)
            fuzzified = updated_memberships**m
        else:
            updated_memberships = polynomial_membership(data, updated, beta)
            fuzzified = polynomial_fuzzifier(updated_memberships, beta)
        objective = float(np.sum(fuzzified * _squared_distances(data, updated)))
        if not np.isfinite(objective) or np.any(~np.isfinite(updated)):
            raise FloatingPointError("fuzzy-clustering update became nonfinite")
        history.append(objective)
        shift = float(np.linalg.norm(updated - centers))
        scale = 1.0 + float(np.linalg.norm(centers))
        centers = updated
        memberships = updated_memberships
        if shift <= tol * scale:
            converged = True
            break

    return _FuzzyCandidate(centers, memberships, history[-1], iteration, converged, history)


def _run_fuzzy(
    data: FloatArray,
    k: int,
    seed: int,
    method: str,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-6,
    m: float = 2.0,
    beta: float = 0.5,
) -> tuple[IntArray, FloatArray, FloatArray, dict[str, Any], bool, str]:
    if n_init < 1 or max_iter < 1 or not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("invalid fuzzy-clustering optimization controls")
    if method == "fcm" and (not np.isfinite(m) or m <= 1.0):
        raise ValueError("m must be finite and larger than one")
    if method == "polynomial_fcm" and (
        not np.isfinite(beta) or not 0.0 <= beta <= POLYNOMIAL_BETA_MAX
    ):
        raise ValueError("beta must lie in [0, 1 - 1e-10]")

    child_seeds = np.random.SeedSequence(seed).spawn(n_init)
    candidates: list[_FuzzyCandidate] = []
    failed_starts = 0
    for child in child_seeds:
        try:
            candidates.append(
                _fit_fuzzy_candidate(
                    data,
                    k,
                    int(child.generate_state(1, dtype=np.uint32)[0]),
                    max_iter,
                    tol,
                    method,
                    m,
                    beta,
                )
            )
        except (FloatingPointError, ValueError):
            failed_starts += 1
    if not candidates:
        raise RuntimeError("all fuzzy-clustering starts failed")
    best = min(candidates, key=lambda candidate: candidate.objective)
    labels = np.argmax(best.memberships, axis=1).astype(np.int64)
    increases = int(np.sum(np.diff(best.objective_history) > 1e-8 * (1.0 + best.objective)))
    diagnostics: dict[str, Any] = {
        "objective": best.objective,
        "iterations": best.iterations,
        "converged": best.converged,
        "n_init": n_init,
        "failed_starts": failed_starts,
        "objective_increases": increases,
        "objective_history": best.objective_history,
    }
    if method == "fcm":
        diagnostics["m"] = float(m)
    else:
        diagnostics["beta"] = float(beta)
        diagnostics["zero_membership_fraction"] = float(np.mean(best.memberships == 0.0))
    message = "converged" if best.converged else "maximum iterations reached"
    return labels, best.centers, best.memberships, diagnostics, best.converged, message


def _centers_from_labels(data: FloatArray, labels: IntArray) -> tuple[IntArray, FloatArray]:
    observed = np.unique(labels[labels >= 0])
    remapped = np.full(labels.shape, -1, dtype=np.int64)
    centers = np.empty((observed.size, data.shape[1]), dtype=float)
    for new_label, old_label in enumerate(observed):
        mask = labels == old_label
        remapped[mask] = new_label
        centers[new_label] = np.mean(data[mask], axis=0)
    return remapped, centers


def _run_sklearn_kmeans(
    data: FloatArray, k: int, seed: int, *, n_init: int = 10, max_iter: int = 300
) -> tuple[IntArray, FloatArray, None, dict[str, Any], bool, str]:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=1e-4,
        algorithm="lloyd",
        random_state=seed,
    ).fit(data)
    diagnostics = {"inertia": float(model.inertia_), "iterations": int(model.n_iter_), "n_init": n_init}
    return model.labels_.astype(np.int64), np.asarray(model.cluster_centers_), None, diagnostics, True, "fit completed"


def _run_ekmeans(
    data: FloatArray,
    k: int,
    seed: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    alpha: float | str = "dvariance",
    scale: float = 2.0,
) -> tuple[IntArray, FloatArray, FloatArray, dict[str, Any], bool, str]:
    try:
        from sklekmeans import EKMeans
    except ImportError as exc:  # pragma: no cover - exercised when optional dependency is absent
        raise ImportError("EKMeans requires the optional 'sklekmeans' package") from exc
    model = EKMeans(
        n_clusters=k,
        alpha=alpha,
        scale=scale,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        use_numba=False,
    ).fit(data)
    memberships = np.asarray(model.U_, dtype=float)
    equilibrium_weights = np.asarray(model.W_, dtype=float)
    if (
        not np.isfinite(model.objective_)
        or not np.isfinite(model.alpha_)
        or np.any(~np.isfinite(equilibrium_weights))
    ):
        raise FloatingPointError("EKMeans returned nonfinite fitted quantities")
    diagnostics = {
        "objective": float(model.objective_),
        # sklekmeans 0.2.1 stores the zero-based loop index rather than the
        # number of completed updates.
        "iterations": int(model.n_iter_) + 1,
        "alpha": float(model.alpha_),
        "n_init": n_init,
        "package": "sklekmeans",
        "negative_equilibrium_weight_fraction": float(np.mean(equilibrium_weights < 0.0)),
    }
    try:
        diagnostics["package_version"] = version("sklekmeans")
    except PackageNotFoundError:  # pragma: no cover - import above already found the package
        diagnostics["package_version"] = "unknown"
    return (
        np.asarray(model.labels_, dtype=np.int64),
        np.asarray(model.cluster_centers_, dtype=float),
        memberships,
        diagnostics,
        True,
        "fit completed",
    )


def _run_hdbscan(
    data: FloatArray,
    k: int,
    seed: int,
    *,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
) -> tuple[IntArray, FloatArray, None, dict[str, Any], bool, str]:
    del seed, k  # HDBSCAN is deterministic and does not take a target k.
    if min_cluster_size is None:
        min_cluster_size = max(5, int(np.ceil(0.02 * data.shape[0])))
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least two")
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        allow_single_cluster=False,
        store_centers="centroid",
        n_jobs=1,
        copy=True,
    ).fit(data)
    original_labels = np.asarray(model.labels_, dtype=np.int64)
    labels, posthoc_centers = _centers_from_labels(data, original_labels)
    centers = np.asarray(model.centroids_, dtype=float) if posthoc_centers.size else posthoc_centers
    found = centers.shape[0]
    diagnostics = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": min_samples,
        "selection_method": "eom",
        "k_used_by_algorithm": False,
        "mean_nonnoise_membership_strength": (
            float(np.mean(model.probabilities_[original_labels >= 0]))
            if np.any(original_labels >= 0)
            else float("nan")
        ),
    }
    message = "fit completed" if found else "fit completed; all observations were labelled noise"
    return labels, centers, None, diagnostics, True, message


def _run_agglomerative(
    data: FloatArray, k: int, seed: int, *, linkage: str = "ward"
) -> tuple[IntArray, FloatArray, None, dict[str, Any], bool, str]:
    del seed
    if linkage not in {"ward", "complete", "average", "single"}:
        raise ValueError("unsupported agglomerative linkage")
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(data)
    labels, centers = _centers_from_labels(data, np.asarray(model.labels_, dtype=np.int64))
    return labels, centers, None, {"linkage": linkage}, True, "fit completed"


def _run_spectral(
    data: FloatArray,
    k: int,
    seed: int,
    *,
    n_neighbors: int = 10,
    n_init: int = 10,
) -> tuple[IntArray, FloatArray, None, dict[str, Any], bool, str]:
    if k >= data.shape[0]:
        raise ValueError("spectral clustering requires k < n")
    neighbors = min(max(1, int(n_neighbors)), data.shape[0] - 1)
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=neighbors,
        assign_labels="kmeans",
        n_init=n_init,
        eigen_solver="arpack",
        n_jobs=1,
        random_state=seed,
    ).fit(data)
    labels, centers = _centers_from_labels(data, np.asarray(model.labels_, dtype=np.int64))
    return (
        labels,
        centers,
        None,
        {"affinity": "nearest_neighbors", "n_neighbors": neighbors, "n_init": n_init},
        True,
        "fit completed",
    )


def _run_gmm_full(
    data: FloatArray,
    k: int,
    seed: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    reg_covar: float | None = None,
) -> tuple[IntArray, FloatArray, FloatArray, dict[str, Any], bool, str]:
    if reg_covar is None:
        reg_covar = max(1e-8, 1e-6 * (1.0 + float(np.mean(np.var(data, axis=0)))))
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        init_params="kmeans",
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=seed,
    ).fit(data)
    covariance_eigenvalues = np.linalg.eigvalsh(np.asarray(model.covariances_, dtype=float))
    if np.any(~np.isfinite(covariance_eigenvalues)) or np.any(covariance_eigenvalues <= 0.0):
        raise FloatingPointError("GMM returned a non-positive-definite covariance")
    memberships = np.asarray(model.predict_proba(data), dtype=float)
    labels = np.argmax(memberships, axis=1).astype(np.int64)
    diagnostics = {
        "lower_bound": float(model.lower_bound_),
        "iterations": int(model.n_iter_),
        "converged": bool(model.converged_),
        "n_init": n_init,
        "reg_covar": float(reg_covar),
        "minimum_covariance_eigenvalue": float(np.min(covariance_eigenvalues)),
        "minimum_component_weight": float(np.min(model.weights_)),
    }
    message = "converged" if model.converged_ else "maximum iterations reached"
    return labels, np.asarray(model.means_), memberships, diagnostics, bool(model.converged_), message


def fit_competitor(
    method: str,
    x: ArrayLike,
    k: int,
    seed: int = 0,
    **options: Any,
) -> ClusteringResult:
    """Fit one competitor and always return a structured success/failure record."""

    canonical = _canonical_method(method)
    start = perf_counter()
    data_for_failure = np.asarray(x)
    n = int(data_for_failure.shape[0]) if data_for_failure.ndim >= 1 else 0
    d = int(data_for_failure.shape[1]) if data_for_failure.ndim == 2 else 0

    try:
        data = _as_data(x)
        if data.shape[0] < 2:
            raise ValueError("competitor fitting requires at least two observations")
        requested_k = _validate_k(k, data.shape[0])
        runners = {
            "fcm": lambda: _run_fuzzy(data, requested_k, seed, "fcm", **options),
            "kmeans": lambda: _run_sklearn_kmeans(data, requested_k, seed, **options),
            "ekmeans": lambda: _run_ekmeans(data, requested_k, seed, **options),
            "polynomial_fcm": lambda: _run_fuzzy(
                data, requested_k, seed, "polynomial_fcm", **options
            ),
            "hdbscan": lambda: _run_hdbscan(data, requested_k, seed, **options),
            "agglomerative": lambda: _run_agglomerative(data, requested_k, seed, **options),
            "spectral": lambda: _run_spectral(data, requested_k, seed, **options),
            "gmm_full": lambda: _run_gmm_full(data, requested_k, seed, **options),
        }
        if canonical not in runners:
            raise ValueError(f"unknown competitor method: {method}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            labels, centers, memberships, diagnostics, success, message = runners[canonical]()
        labels = np.asarray(labels, dtype=np.int64)
        centers = np.asarray(centers, dtype=float).reshape(-1, data.shape[1])
        if labels.shape != (data.shape[0],) or np.any(~np.isfinite(centers)):
            raise FloatingPointError("competitor returned malformed labels or centers")
        if memberships is not None:
            memberships = np.asarray(memberships, dtype=float)
            if (
                memberships.shape[0] != data.shape[0]
                or np.any(~np.isfinite(memberships))
                or np.any(memberships < -1e-12)
                or not np.allclose(memberships.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
            ):
                raise FloatingPointError("competitor returned malformed memberships")
        diagnostics = dict(diagnostics)
        diagnostics["warnings"] = [str(item.message) for item in caught]
        occupied = int(np.unique(labels[labels >= 0]).size)
        if np.any(labels >= centers.shape[0]):
            raise FloatingPointError("competitor returned labels inconsistent with its centers")
        if memberships is not None and memberships.shape[1] != centers.shape[0]:
            raise FloatingPointError("competitor returned memberships inconsistent with its centers")
        diagnostics["occupied_clusters"] = occupied
        diagnostics["n_centers_returned"] = int(centers.shape[0])
        return ClusteringResult(
            method=canonical,
            labels=labels,
            centers=centers,
            memberships=memberships,
            runtime_seconds=perf_counter() - start,
            success=bool(success),
            message=message,
            requested_k=requested_k,
            n_clusters_found=occupied,
            noise_fraction=float(np.mean(labels < 0)),
            seed=int(seed),
            diagnostics=diagnostics,
        )
    except Exception as exc:
        return ClusteringResult(
            method=canonical,
            labels=np.full(n, -1, dtype=np.int64),
            centers=np.empty((0, d), dtype=float),
            memberships=None,
            runtime_seconds=perf_counter() - start,
            success=False,
            message=f"{type(exc).__name__}: {exc}",
            requested_k=int(k) if isinstance(k, (int, np.integer)) else -1,
            n_clusters_found=0,
            noise_fraction=1.0,
            seed=int(seed),
            diagnostics={"exception_type": type(exc).__name__},
        )


def fit_competitors(
    x: ArrayLike,
    k: int,
    methods: Sequence[str] = DEFAULT_METHODS,
    seed: int = 0,
    method_options: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, ClusteringResult]:
    """Fit several methods with stable method-specific child seeds."""

    options = method_options or {}
    children = np.random.SeedSequence(seed).spawn(len(methods))
    results: dict[str, ClusteringResult] = {}
    for method, child in zip(methods, children, strict=True):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        canonical = _canonical_method(method)
        method_specific = options.get(method, options.get(canonical, {}))
        result = fit_competitor(method, x, k, seed=child_seed, **dict(method_specific))
        results[result.method] = result
    return results


__all__ = [
    "ClusteringResult",
    "DEFAULT_METHODS",
    "fcm_membership",
    "fit_competitor",
    "fit_competitors",
    "polynomial_fuzzifier",
    "polynomial_membership",
]
