"""Reviewer-driven inference experiments for the WFCM revision.

This script contains three deliberately separate experiments.

1. ``coverage`` compares model-Fisher Wald intervals with full-refit
   nonparametric percentile-bootstrap intervals under a correctly specified,
   separated one-dimensional WFCM model.
2. ``lrt`` studies a bootstrap-calibrated implemented approximate
   likelihood-ratio statistic for pairwise center coalescence.  Its collapsed
   one-center null is fitted by the analytic Gaussian MLE; the unrestricted
   endpoint is still produced by fixed-cloud finite-IS multistart
   optimization, with the analytic null's exact coincident-center embedding
   retained explicitly in the final scored candidate set.  Coalescence is
   nonregular, so the ordinary chi-square p-value is retained only as a
   diagnostic.  This implemented statistic is distinct from the ideal exact
   global likelihood-ratio statistic used in the theory.
3. ``selection`` reports both conditional-on-selecting-the-true-k coverage and
   unconditional select-and-cover probability after weighted-XB selection.

The one-dimensional design is intentional: samples from the WFCM density are
independent numerical inverse-CDF draws, so the experiment does not inherit
the serial dependence of the exploratory notebook's single MCMC chain.  Every
reported fit nevertheless uses the same finite-importance-sampling estimator
as the general implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import chi2, norm

from simulations.wfcm_core import (
    ISContext,
    WFCMParameterization,
    WFCMParams,
    align_to_reference,
    average_nll_and_gradient,
    build_proposal,
    energy,
    energy_and_gradient_free,
    fit_wfcm,
    sample_wfcm_1d,
    weighted_xie_beni,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "revision_work" / "results" / "inference"
PRIMARY_LRT_NAME = (
    "bootstrap-calibrated implemented approximate likelihood-ratio statistic"
)
LRT_PROTOCOL_VERSION = "analytic-gaussian-null-explicit-candidate-v1"
LRT_NULL_ESTIMATOR = "analytic_gaussian_mle"
LRT_NULL_SCORING_METHOD = "closed_form_gaussian_loglik"
LRT_ALTERNATIVE_CANDIDATE_RULE = (
    "max_fitted_alternative_and_embedded_analytic_null"
)
SELECTION_PROTOCOL_VERSION = "wxbi-k234-fixed-m2-shared-fisher-v1"
SELECTION_CANDIDATE_K = (2, 3, 4)
SELECTION_CANDIDATE_SEED_PATHS = {2: (22,), 3: (23, 1), 4: (23, 2)}
SELECTION_TRUE_K = 2
SELECTION_TRUE_SIGMA = 1.0
SELECTION_TRUE_CENTERS = (-2.0, 2.0)
SELECTION_TRUE_WEIGHTS = (0.3, 0.7)
SELECTION_FIXED_M = 2.0
SELECTION_IMPORTANCE_MULTIPLIER = 12.0
SELECTION_MINIMUM_IMPORTANCE_SIZE = 4_000


@dataclass(frozen=True)
class Profile:
    coverage_reps: int
    bootstrap_reps: int
    lrt_reps: int
    lrt_bootstrap_reps: int
    selection_reps: int
    coverage_ns: tuple[int, ...]
    lrt_ns: tuple[int, ...]
    selection_ns: tuple[int, ...]
    fit_n_init: int
    maxiter: int
    fisher_draws: int


@dataclass(frozen=True)
class AnalyticNullFit:
    """Analytic one-center null plus its value on the alternative fit cloud.

    ``average_nll`` is deliberately retained as a compatibility attribute for
    the existing structured diagnostics.  It is the finite-cloud average NLL
    of ``embedded_params`` in the *two-center* parameterization on the exact
    same cloud used to optimize the alternative; it is not the objective of a
    numerical one-center fit.
    """

    params: WFCMParams
    embedded_params: WFCMParams
    exact_loglik: float
    average_nll: float
    success: bool = True
    optimizer_success: bool = False
    gradient_inf_norm: float = 0.0
    message: str = "analytic Gaussian restricted MLE; no null optimizer used"
    estimator_name: str = LRT_NULL_ESTIMATOR


PROFILES = {
    "smoke": Profile(2, 5, 2, 5, 2, (100,), (100,), (100,), 2, 180, 5_000),
    "pilot": Profile(12, 39, 12, 39, 20, (200, 500), (200,), (150, 300), 2, 300, 20_000),
    # Coverage Monte Carlo SE is at most 1.6 percentage points near 95%; the
    # LRT rejection-rate SE is at most 1.8 points at the 5% boundary.
    "full": Profile(200, 199, 150, 199, 250, (200, 500), (200, 500), (150, 300), 3, 450, 50_000),
}


def importance_size(n: int, multiplier: float = 20.0) -> int:
    """A finite-sample implementation of an M/n-large regime.

    The default M=20n does not pretend to establish first-order equivalence;
    that distinction is handled explicitly in the theory and computation
    experiment.  It keeps Monte Carlo noise small enough for the interval
    comparison while retaining the actual finite-IS estimator.
    """

    return max(4_000, int(np.ceil(multiplier * n)))


def _spawn_seed(seed: int, *indices: int) -> int:
    sequence = np.random.SeedSequence([int(seed), *(int(i) for i in indices)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _log_normalizer_grid_1d(params: WFCMParams, grid_size: int = 100_001) -> float:
    """High-accuracy deterministic 1D normalizer used only for LRT scoring.

    The base grid controls the tails and the interaction region.  If that
    grid is too coarse relative to a center's local Gaussian scale, a dense
    local grid is added around that center and the trapezoid rule is applied
    on the resulting nonuniform union.  This matters for admissible fits with
    a very small scale and widely separated centers: a global uniform grid can
    otherwise skip a narrow mode entirely.
    """

    params.validate()
    if params.d != 1:
        raise ValueError("grid normalizer is only available in one dimension")
    if (
        isinstance(grid_size, (bool, np.bool_))
        or not isinstance(grid_size, (int, np.integer))
        or grid_size < 1_001
    ):
        raise ValueError("grid_size must be an integer of at least 1001")
    a = 1.0 / (params.m - 1.0)
    tail_coefficient = (
        np.sum(params.weights ** (-a)) ** (-(params.m - 1.0)) / params.sigma**2
    )
    centers = params.centers[:, 0]
    # With one center, or with exactly coincident centers, the energy is
    # tail_coefficient * (x-v)^2 everywhere, not merely in the tails.
    if np.all(centers == centers[0]):
        return float(0.5 * (np.log(np.pi) - np.log(tail_coefficient)))

    asymptotic_sd = np.sqrt(1.0 / (2.0 * tail_coefficient))
    span = max(float(np.ptp(centers)), asymptotic_sd)
    lo = float(np.min(centers) - 12.0 * asymptotic_sd - span)
    hi = float(np.max(centers) + 12.0 * asymptotic_sd + span)
    base_grid = np.linspace(lo, hi, grid_size)
    base_dx = (hi - lo) / (grid_size - 1)
    grids = [base_grid]
    # E(v_j+r)=w_j*r^2/sigma^2+o(r^2), so this is the local
    # standard-deviation scale of the normalized quadratic approximation.
    local_sds = params.sigma / np.sqrt(2.0 * params.weights)
    for center, local_sd in zip(centers, local_sds, strict=True):
        if base_dx > local_sd / 32.0:
            left = max(lo, float(center - 12.0 * local_sd))
            right = min(hi, float(center + 12.0 * local_sd))
            # 2048 intervals give at least 85 points per local standard
            # deviation and make the local trapezoid error negligible on the
            # likelihood-ratio scale.
            grids.append(np.linspace(left, right, 2_049))
    grid = np.unique(np.concatenate(grids))
    log_kernel = -energy(
        grid[:, None], params.sigma, params.centers, params.weights, params.m
    )
    widths = np.diff(grid)
    log_segment_integrals = (
        np.log(widths)
        + np.logaddexp(log_kernel[:-1], log_kernel[1:])
        - np.log(2.0)
    )
    return float(logsumexp(log_segment_integrals))


def exact_loglik_1d(x: np.ndarray, params: WFCMParams) -> float:
    """Log likelihood with a deterministic high-accuracy 1D normalizer."""

    values = np.asarray(x, dtype=float).reshape(-1, 1)
    return float(-np.sum(energy(values, params.sigma, params.centers, params.weights, params.m))
                 - values.shape[0] * _log_normalizer_grid_1d(params))


def normal_null_fit(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, MLE standard deviation, maximized log likelihood)."""

    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size < 2 or np.any(~np.isfinite(values)):
        raise ValueError("analytic Gaussian null MLE needs at least two finite values")
    mean = float(np.mean(values))
    sd = float(np.sqrt(np.mean((values - mean) ** 2)))
    # For a constant sample the unrestricted Gaussian likelihood has no
    # finite interior maximizer.  Silently flooring the scale would no longer
    # be an analytic restricted MLE, so expose the failure to the structured
    # outer-analysis record instead.
    if not np.isfinite(sd) or sd <= 0.0:
        raise ValueError("analytic Gaussian null MLE has zero or nonfinite scale")
    loglik = float(np.sum(norm.logpdf(values, loc=mean, scale=sd)))
    if not np.isfinite(loglik):
        raise FloatingPointError("analytic Gaussian null log likelihood is nonfinite")
    return mean, sd, loglik


def _null_embedded_start(
    x: np.ndarray,
    m: float,
    null_params: WFCMParams | None = None,
    split_fraction: float = 0.0,
) -> WFCMParams:
    """Represent a k=1 fit exactly by coincident centers in the k=2 model."""

    mean, sd, _ = normal_null_fit(x)
    if null_params is not None:
        if null_params.k != 1 or null_params.d != 1:
            raise ValueError("the restricted fit must have k=d=1")
        mean = float(null_params.centers[0, 0])
        sigma_k1 = float(null_params.sigma)
        sd = sigma_k1 / np.sqrt(2.0)
    else:
        sigma_k1 = np.sqrt(2.0) * sd
    epsilon = max(split_fraction * sd, 0.0)
    # A k=1 WFCM has sigma=sqrt(2)*sd.  With two coincident equal-weight
    # centers and m=2, rho is divided by four, hence sigma is divided by two.
    # The expression below also covers fixed m != 2 through B=sum w^{-a}.
    weights = np.array([0.5, 0.5])
    a = 1.0 / (m - 1.0)
    b = float(np.sum(weights ** (-a)))
    # rho at the coincident pair is r^2/B^(m-1), where
    # B=sum_j w_j^{-1/(m-1)}.
    sigma_k2 = sigma_k1 / b ** ((m - 1.0) / 2.0)
    return WFCMParams(
        sigma=float(sigma_k2),
        centers=np.array([[mean - epsilon], [mean + epsilon]]),
        weights=weights,
        m=m,
    )


def fit_alternative_for_lrt(
    x: np.ndarray,
    *,
    m: float,
    seed: int,
    m_is: int,
    n_init: int,
    maxiter: int,
) -> tuple[object, AnalyticNullFit, float, float]:
    """Fit the numerical alternative and score it against the analytic null.

    Returns the finite-IS two-center endpoint, the analytic one-center fit
    (including its exact embedding evaluated on the alternative cloud), the
    nonnegative explicit-candidate statistic, and the unselected
    endpoint-minus-null raw statistic, respectively.
    """

    values = np.asarray(x, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    proposal = build_proposal(values, 2, "defensive", int(rng.integers(0, 2**31 - 1)))
    context = ISContext.draw(proposal, m_is, int(rng.integers(0, 2**31 - 1)))
    mean, sd, null_loglik = normal_null_fit(values)
    analytic_null_params = WFCMParams(
        sigma=np.sqrt(2.0) * sd,
        centers=np.array([[mean]]),
        weights=np.array([1.0]),
        m=m,
    )
    embedded_null = _null_embedded_start(values, m, analytic_null_params)
    alternative_spec = WFCMParameterization(2, 1, m_fixed=m)
    embedded_free = alternative_spec.pack(embedded_null)
    embedded_bounds = alternative_spec.bounds(values)
    if any(
        coordinate < lower or coordinate > upper
        for coordinate, (lower, upper) in zip(
            embedded_free, embedded_bounds, strict=True
        )
    ):
        raise ValueError(
            "analytic null embedding lies outside the alternative fit bounds"
        )
    embedded_null_average_nll, _ = average_nll_and_gradient(
        embedded_free,
        values,
        context=context,
        spec=alternative_spec,
    )
    if not np.isfinite(embedded_null_average_nll):
        raise FloatingPointError(
            "analytic null embedding has nonfinite finite-cloud objective"
        )
    null_fit = AnalyticNullFit(
        params=analytic_null_params,
        embedded_params=embedded_null,
        exact_loglik=null_loglik,
        average_nll=float(embedded_null_average_nll),
    )
    fit = fit_wfcm(
        values,
        k=2,
        m=m,
        importance_size=m_is,
        proposal_kind="defensive",
        n_init=n_init,
        maxiter=maxiter,
        seed=int(rng.integers(0, 2**31 - 1)),
        initial_params=embedded_null,
        context=context,
    )
    # The restricted denominator is the exact analytic Gaussian maximum.  The
    # alternative endpoint is scored by deterministic high-accuracy
    # quadrature, separately from the finite-IS objective used to find it.
    # Explicitly maximizing over that endpoint and the exact coincident-center
    # embedding makes scored nesting a property of the candidate-set
    # definition, not an inference from an optimizer start.  The unselected
    # endpoint-minus-null difference is retained for diagnosis.
    alternative_loglik = exact_loglik_1d(values, fit.params)
    raw = 2.0 * (alternative_loglik - null_loglik)
    if not np.isfinite(raw):
        raise FloatingPointError("nonfinite deterministic LRT scoring result")
    selected_loglik = max(alternative_loglik, null_loglik)
    statistic = 2.0 * (selected_loglik - null_loglik)
    # Avoid a negative signed zero or a few ulps from the subtraction above;
    # the equality is also independently checked by the reporting pipeline.
    statistic = max(0.0, float(statistic))
    return fit, null_fit, statistic, float(raw)


def fisher_covariance_from_model_draws(
    fit,
    *,
    draws: int,
    seed: int,
    eigen_floor: float = 1e-7,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate inverse Fisher information from independent fitted-model draws."""

    spec = fit.parameterization
    if spec is None:
        spec = WFCMParameterization(
            k=fit.params.k,
            d=1,
            m_fixed=fit.params.m,
            weight_floor=1e-4,
        )
    if spec.k != fit.params.k or spec.d != 1 or spec.m_fixed != fit.params.m:
        raise ValueError("fit and stored parameterization are incompatible")
    model_x = sample_wfcm_1d(fit.params, draws, seed, grid_size=100_001)
    _, gradients = energy_and_gradient_free(fit.free_params, model_x, spec)
    centered = gradients - np.mean(gradients, axis=0)
    information = centered.T @ centered / draws
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    largest = float(max(np.max(eigenvalues), 1.0))
    floor = eigen_floor * largest
    clipped = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    diagnostics = {
        "fisher_min_eigenvalue": float(np.min(eigenvalues)),
        "fisher_max_eigenvalue": float(np.max(eigenvalues)),
        "fisher_condition_clipped": float(np.max(clipped) / np.min(clipped)),
        "fisher_eigen_floor": float(floor),
    }
    return inverse, diagnostics


def fisher_center_intervals(
    fit,
    n: int,
    *,
    draws: int,
    seed: int,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    inverse, diagnostics = fisher_covariance_from_model_draws(
        fit, draws=draws, seed=seed
    )
    spec = fit.parameterization
    if spec is None:
        spec = WFCMParameterization(fit.params.k, 1, m_fixed=fit.params.m)
    center_indices = np.arange(spec.center_slice.start, spec.center_slice.stop)
    standard_errors = np.sqrt(np.maximum(np.diag(inverse)[center_indices] / n, 0.0))
    critical = float(norm.ppf(0.5 + level / 2.0))
    estimates = fit.params.centers[:, 0]
    return estimates - critical * standard_errors, estimates + critical * standard_errors, diagnostics


class BootstrapIntervalFailure(RuntimeError):
    """Bootstrap interval failure that retains the attempted-refit counts."""

    def __init__(self, message: str, diagnostics: dict[str, float]):
        super().__init__(message)
        self.diagnostics = diagnostics


def bootstrap_center_intervals(
    x: np.ndarray,
    fit,
    *,
    bootstrap_reps: int,
    seed: int,
    m_is: int,
    n_init: int,
    maxiter: int,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = np.asarray(x, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    aligned = []
    finite = 0
    start = perf_counter()
    for b in range(bootstrap_reps):
        try:
            sample = values[
                rng.integers(0, values.shape[0], size=values.shape[0])
            ]
            bootstrap_seed = _spawn_seed(seed, b, 771)
            boot_fit = fit_wfcm(
                sample,
                k=fit.params.k,
                m=fit.params.m,
                importance_size=m_is,
                proposal_kind="defensive",
                n_init=n_init,
                maxiter=maxiter,
                seed=bootstrap_seed,
                initial_params=fit.params,
            )
            centers, _, _ = align_to_reference(
                boot_fit.params.centers,
                boot_fit.params.weights,
                fit.params.centers,
            )
            if np.all(np.isfinite(centers)):
                finite += 1
                # A finite endpoint from a failed local optimization is not a
                # valid bootstrap replicate and must not enter the quantiles.
                if boot_fit.success:
                    aligned.append(centers[:, 0])
        except Exception:
            # One failed refit is part of the attempted bootstrap denominator;
            # it must not abort the remaining prespecified refits.
            continue
    elapsed = perf_counter() - start
    diagnostics = {
        "bootstrap_requested": float(bootstrap_reps),
        "bootstrap_finite": float(finite),
        "bootstrap_valid": float(len(aligned)),
        "bootstrap_optimizer_success": float(len(aligned)),
        "bootstrap_runtime_seconds": float(elapsed),
    }
    if len(aligned) < max(5, int(np.ceil(0.8 * bootstrap_reps))):
        raise BootstrapIntervalFailure(
            "fewer than 80% convergence-valid bootstrap refits",
            diagnostics,
        )
    draws = np.asarray(aligned)
    alpha = 1.0 - level
    try:
        lower = np.quantile(draws, alpha / 2.0, axis=0)
        upper = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    except Exception as exc:
        raise BootstrapIntervalFailure(
            f"bootstrap endpoint calculation failed: {type(exc).__name__}: {exc}",
            diagnostics,
        ) from exc
    return lower, upper, diagnostics


def _coverage_one(profile: Profile, n: int, replicate: int, base_seed: int) -> list[dict]:
    truth = WFCMParams(
        sigma=0.9,
        centers=np.array([[-2.0], [2.0]]),
        weights=np.array([0.35, 0.65]),
        m=2.0,
    )
    data_seed = _spawn_seed(base_seed, 1, n, replicate)
    fit_seed = _spawn_seed(base_seed, 2, n, replicate)
    m_is = importance_size(n)
    fit = None
    values = None
    aligned_centers = np.full_like(truth.centers, np.nan)
    estimated_for_truth = np.arange(truth.k)
    fit_success = False
    optimizer_success = False
    gradient_inf_norm = np.nan
    ess = np.nan
    ess_fraction = np.nan
    base_runtime = 0.0
    base_failure_message = ""
    base_exception_type = ""

    setup_start = perf_counter()
    try:
        values = sample_wfcm_1d(truth, n, data_seed)
        fit_start = perf_counter()
        try:
            fit = fit_wfcm(
                values,
                k=2,
                m=truth.m,
                importance_size=m_is,
                proposal_kind="defensive",
                n_init=profile.fit_n_init,
                maxiter=profile.maxiter,
                seed=fit_seed,
            )
        finally:
            # Preserve the original successful-path definition: base runtime
            # times the observed fit, not data generation.
            base_runtime = perf_counter() - fit_start
        fit_success = bool(fit.success)
        optimizer_success = bool(fit.optimizer_success)
        gradient_inf_norm = float(fit.gradient_inf_norm)
        ess = float(fit.diagnostics["ess"])
        ess_fraction = float(fit.diagnostics["ess_fraction"])
        if not fit_success:
            base_failure_message = (
                str(getattr(fit, "message", ""))
                or "observed coverage fit did not satisfy convergence checks"
            )
        else:
            aligned_centers, _, estimated_for_truth = align_to_reference(
                fit.params.centers, fit.params.weights, truth.centers
            )
            if not np.all(np.isfinite(aligned_centers)):
                raise FloatingPointError("observed aligned centers are nonfinite")
    except Exception as exc:
        if base_runtime == 0.0:
            base_runtime = perf_counter() - setup_start
        base_failure_message = f"{type(exc).__name__}: {exc}"
        base_exception_type = type(exc).__name__

    fisher_lo = np.full(truth.k, np.nan)
    fisher_hi = np.full(truth.k, np.nan)
    fisher_diag = {
        "fisher_min_eigenvalue": np.nan,
        "fisher_max_eigenvalue": np.nan,
        "fisher_condition_clipped": np.nan,
        "fisher_eigen_floor": np.nan,
    }
    fisher_runtime = 0.0
    fisher_success = False
    fisher_failure_message = base_failure_message
    fisher_exception_type = base_exception_type
    if fit_success and not base_failure_message:
        fisher_start = perf_counter()
        try:
            fisher_lo_raw, fisher_hi_raw, fisher_diag = fisher_center_intervals(
                fit,
                n,
                draws=profile.fisher_draws,
                seed=_spawn_seed(base_seed, 3, n, replicate),
            )
            fisher_lo = np.asarray(fisher_lo_raw, dtype=float)[estimated_for_truth]
            fisher_hi = np.asarray(fisher_hi_raw, dtype=float)[estimated_for_truth]
            fisher_success = bool(
                np.all(np.isfinite(fisher_lo))
                and np.all(np.isfinite(fisher_hi))
                and np.all(fisher_hi >= fisher_lo)
            )
            if not fisher_success:
                raise FloatingPointError("Fisher interval endpoints are nonfinite or unordered")
            fisher_failure_message = ""
            fisher_exception_type = ""
        except Exception as exc:
            fisher_success = False
            fisher_failure_message = f"{type(exc).__name__}: {exc}"
            fisher_exception_type = type(exc).__name__
        finally:
            fisher_runtime = perf_counter() - fisher_start

    bootstrap_lo = np.full(truth.k, np.nan)
    bootstrap_hi = np.full(truth.k, np.nan)
    bootstrap_diag = {
        "bootstrap_requested": float(profile.bootstrap_reps),
        "bootstrap_finite": 0.0,
        "bootstrap_valid": 0.0,
        "bootstrap_optimizer_success": 0.0,
        "bootstrap_runtime_seconds": 0.0,
    }
    bootstrap_success = False
    bootstrap_failure_message = base_failure_message
    bootstrap_exception_type = base_exception_type
    if fit_success and not base_failure_message:
        bootstrap_start = perf_counter()
        try:
            bootstrap_lo_raw, bootstrap_hi_raw, bootstrap_diag = (
                bootstrap_center_intervals(
                    values,
                    fit,
                    bootstrap_reps=profile.bootstrap_reps,
                    seed=_spawn_seed(base_seed, 4, n, replicate),
                    m_is=m_is,
                    # A full-refit bootstrap must repeat the observed fitting
                    # rule rather than silently reducing optimization starts.
                    n_init=profile.fit_n_init,
                    maxiter=profile.maxiter,
                )
            )
            bootstrap_lo = np.asarray(bootstrap_lo_raw, dtype=float)[estimated_for_truth]
            bootstrap_hi = np.asarray(bootstrap_hi_raw, dtype=float)[estimated_for_truth]
            bootstrap_success = bool(
                np.all(np.isfinite(bootstrap_lo))
                and np.all(np.isfinite(bootstrap_hi))
                and np.all(bootstrap_hi >= bootstrap_lo)
                and bootstrap_diag["bootstrap_valid"]
                >= max(5, int(np.ceil(0.8 * profile.bootstrap_reps)))
            )
            if not bootstrap_success:
                raise FloatingPointError(
                    "bootstrap interval endpoints/counts are invalid"
                )
            bootstrap_failure_message = ""
            bootstrap_exception_type = ""
        except Exception as exc:
            bootstrap_success = False
            if isinstance(exc, BootstrapIntervalFailure):
                bootstrap_diag = exc.diagnostics
            else:
                bootstrap_diag["bootstrap_runtime_seconds"] = (
                    perf_counter() - bootstrap_start
                )
            bootstrap_failure_message = f"{type(exc).__name__}: {exc}"
            bootstrap_exception_type = type(exc).__name__

    common = {
        "experiment": "coverage",
        "n": n,
        "replicate": replicate,
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "importance_size": m_is,
        "fit_success": fit_success,
        "optimizer_success": optimizer_success,
        "gradient_inf_norm": gradient_inf_norm,
        "ess": ess,
        "ess_fraction": ess_fraction,
        "base_fit_runtime_seconds": base_runtime,
        "fisher_runtime_seconds": fisher_runtime,
        **fisher_diag,
        **bootstrap_diag,
    }
    rows = []
    for method, lower, upper, method_success, failure_message, exception_type in (
        (
            "fisher", fisher_lo, fisher_hi, fisher_success,
            fisher_failure_message, fisher_exception_type,
        ),
        (
            "bootstrap", bootstrap_lo, bootstrap_hi, bootstrap_success,
            bootstrap_failure_message, bootstrap_exception_type,
        ),
    ):
        for center in range(2):
            covered = (
                bool(lower[center] <= truth.centers[center, 0] <= upper[center])
                if method_success
                else None
            )
            rows.append(
                {
                    **common,
                    "method": method,
                    "center": center,
                    "truth": float(truth.centers[center, 0]),
                    "estimate": float(aligned_centers[center, 0]),
                    "lower": float(lower[center]),
                    "upper": float(upper[center]),
                    "covered": covered,
                    "length": float(upper[center] - lower[center]),
                    "method_success": method_success,
                    "failure_message": failure_message,
                    "exception_type": exception_type,
                }
            )
    return rows


def _lrt_one(
    profile: Profile,
    n: int,
    separation: float,
    replicate: int,
    base_seed: int,
) -> dict:
    m = 2.0
    sigma = 1.0 / np.sqrt(2.0)
    data_seed = _spawn_seed(base_seed, 11, n, int(100 * separation), replicate)
    m_is = importance_size(n, multiplier=15.0)
    fit_seed = _spawn_seed(base_seed, 12, n, int(100 * separation), replicate)
    # These are diagnostics, not truncation rules.  Define them even on a
    # failed outer analysis so every scheduled replicate has the same schema.
    negative_tolerance = 1e-7 * n
    fitting_objective_negative_tolerance = 1e-8 * n

    values = None
    fit = None
    null_fit = None
    fit_success = False
    null_fit_success = False
    statistic = np.nan
    raw_statistic = np.nan
    fitting_objective_raw_statistic = np.nan
    analytic_null_loglik = np.nan
    fitted_alternative_loglik = np.nan
    analytic_null_mean = np.nan
    analytic_null_normal_sd = np.nan
    analytic_null_wfcm_sigma = np.nan
    scored_candidate_selected = None
    observed_runtime = 0.0
    bootstrap_runtime = 0.0
    failure_message = ""
    exception_type = ""
    observed_valid = False

    data_start = perf_counter()
    try:
        if separation == 0.0:
            rng = np.random.default_rng(data_seed)
            values = rng.normal(0.0, 1.0, size=(n, 1))
        else:
            truth = WFCMParams(
                sigma=sigma,
                centers=np.array([[-separation / 2.0], [separation / 2.0]]),
                weights=np.array([0.5, 0.5]),
                m=m,
            )
            values = sample_wfcm_1d(truth, n, data_seed)
    except Exception as exc:
        observed_runtime = perf_counter() - data_start
        failure_message = f"{type(exc).__name__}: {exc}"
        exception_type = type(exc).__name__

    if values is not None:
        observed_start = perf_counter()
        try:
            fit, null_fit, statistic, raw_statistic = fit_alternative_for_lrt(
                values,
                m=m,
                seed=fit_seed,
                m_is=m_is,
                n_init=profile.fit_n_init,
                maxiter=profile.maxiter,
            )
            fit_success = bool(fit.success)
            null_fit_success = bool(null_fit.success)
            analytic_null_loglik = float(null_fit.exact_loglik)
            fitted_alternative_loglik = float(
                analytic_null_loglik + raw_statistic / 2.0
            )
            analytic_null_mean = float(null_fit.params.centers[0, 0])
            analytic_null_wfcm_sigma = float(null_fit.params.sigma)
            analytic_null_normal_sd = float(
                analytic_null_wfcm_sigma / np.sqrt(2.0)
            )
            scored_candidate_selected = (
                "embedded_analytic_null"
                if raw_statistic < 0.0
                else "fitted_alternative"
            )
            # On the common fixed importance cloud, -n times average_nll is
            # the implemented fitting log likelihood.  Keep it separate from
            # raw_statistic, which deterministically re-scores the fits.
            fitting_objective_raw_statistic = float(
                2.0 * n * (null_fit.average_nll - fit.average_nll)
            )
            observed_valid = bool(
                fit_success
                and null_fit_success
                and np.isfinite(statistic)
                and np.isfinite(raw_statistic)
                and np.isfinite(fitting_objective_raw_statistic)
            )
            if not observed_valid:
                messages = []
                if not fit_success:
                    messages.append(
                        "alternative fit: "
                        + (
                            str(getattr(fit, "message", ""))
                            or "did not satisfy convergence checks"
                        )
                    )
                if not null_fit_success:
                    messages.append(
                        "null fit: "
                        + (
                            str(getattr(null_fit, "message", ""))
                            or "did not satisfy convergence checks"
                        )
                    )
                if not messages:
                    messages.append("observed LRT statistics are nonfinite")
                failure_message = "; ".join(messages)
        except Exception as exc:
            failure_message = f"{type(exc).__name__}: {exc}"
            exception_type = type(exc).__name__
        finally:
            observed_runtime = perf_counter() - observed_start

    bootstrap_statistics = []
    bootstrap_raw_valid = []
    bootstrap_fitting_objective_raw_valid = []
    bootstrap_finite = 0
    first_bootstrap_exception_message = ""
    first_bootstrap_exception_type = ""
    if observed_valid:
        bootstrap_start = perf_counter()
        try:
            null_mean = float(null_fit.params.centers[0, 0])
            null_sd = float(null_fit.params.sigma / np.sqrt(2.0))
            rng = np.random.default_rng(
                _spawn_seed(
                    base_seed, 13, n, int(100 * separation), replicate
                )
            )
            for b in range(profile.lrt_bootstrap_reps):
                bootstrap_x = rng.normal(null_mean, null_sd, size=(n, 1))
                try:
                    boot_fit, boot_null_fit, boot_stat, boot_raw = (
                        fit_alternative_for_lrt(
                            bootstrap_x,
                            m=m,
                            seed=_spawn_seed(
                                base_seed,
                                14,
                                n,
                                int(100 * separation),
                                replicate,
                                b,
                            ),
                            m_is=m_is,
                            # Repeat the observed fitting rule inside each
                            # null bootstrap sample.
                            n_init=profile.fit_n_init,
                            maxiter=profile.maxiter,
                        )
                    )
                    if np.isfinite(boot_stat) and np.isfinite(boot_raw):
                        bootstrap_finite += 1
                        boot_fitting_raw = float(
                            2.0
                            * n
                            * (
                                boot_null_fit.average_nll
                                - boot_fit.average_nll
                            )
                        )
                        if (
                            boot_fit.success
                            and boot_null_fit.success
                            and np.isfinite(boot_fitting_raw)
                        ):
                            bootstrap_statistics.append(float(boot_stat))
                            bootstrap_raw_valid.append(float(boot_raw))
                            bootstrap_fitting_objective_raw_valid.append(
                                boot_fitting_raw
                            )
                except Exception as exc:
                    if not first_bootstrap_exception_message:
                        first_bootstrap_exception_message = (
                            f"{type(exc).__name__}: {exc}"
                        )
                        first_bootstrap_exception_type = type(exc).__name__
                    continue
        except Exception as exc:
            failure_message = f"{type(exc).__name__}: {exc}"
            exception_type = type(exc).__name__
        finally:
            bootstrap_runtime = perf_counter() - bootstrap_start

    boot_array = np.asarray(bootstrap_statistics, dtype=float)
    boot_raw_array = np.asarray(bootstrap_raw_valid, dtype=float)
    boot_fitting_raw_array = np.asarray(
        bootstrap_fitting_objective_raw_valid, dtype=float
    )
    minimum_valid = max(
        5, int(np.ceil(0.8 * profile.lrt_bootstrap_reps))
    )
    enough_bootstrap = len(bootstrap_statistics) >= minimum_valid
    bootstrap_p = np.nan
    chi_square_p = (
        float(chi2.sf(statistic, df=1)) if np.isfinite(statistic) else np.nan
    )
    if observed_valid and enough_bootstrap and not failure_message:
        bootstrap_p = float(
            (1.0 + np.sum(boot_array >= statistic))
            / (len(boot_array) + 1.0)
        )
    analysis_success = bool(
        observed_valid
        and enough_bootstrap
        and np.isfinite(bootstrap_p)
        and np.isfinite(chi_square_p)
    )
    if not analysis_success and not failure_message:
        failure_message = (
            "fewer than 80% convergence-valid parametric-bootstrap LRT "
            f"refits ({len(bootstrap_statistics)}/{profile.lrt_bootstrap_reps})"
        )
        if first_bootstrap_exception_message:
            failure_message += "; first refit exception: " + first_bootstrap_exception_message
            exception_type = first_bootstrap_exception_type
    if analysis_success:
        failure_message = ""
        exception_type = ""

    raw_defined = bool(np.isfinite(raw_statistic))
    fitting_raw_defined = bool(np.isfinite(fitting_objective_raw_statistic))
    bootstrap_raw_defined = boot_raw_array.size > 0
    bootstrap_fitting_raw_defined = boot_fitting_raw_array.size > 0
    bootstrap_cross_defined = bool(
        bootstrap_raw_defined
        and bootstrap_fitting_raw_defined
        and boot_raw_array.size == boot_fitting_raw_array.size
    )
    if bootstrap_cross_defined:
        bootstrap_cross = (
            (boot_raw_array < 0.0)
            & (
                boot_fitting_raw_array
                >= -fitting_objective_negative_tolerance
            )
        )
        bootstrap_material_cross = (
            (boot_raw_array < -negative_tolerance)
            & (
                boot_fitting_raw_array
                >= -fitting_objective_negative_tolerance
            )
        )
    else:
        bootstrap_cross = np.asarray([], dtype=bool)
        bootstrap_material_cross = np.asarray([], dtype=bool)
    return {
        "experiment": "lrt",
        "statistic_name": PRIMARY_LRT_NAME,
        "lrt_protocol_version": LRT_PROTOCOL_VERSION,
        "n": n,
        "separation": separation,
        "replicate": replicate,
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "importance_size": m_is,
        "scoring_method": "deterministic_1d_quadrature",
        "scoring_grid_size": 100_001,
        "scoring_base_grid_size": 100_001,
        "scoring_adaptive_local_refinement": True,
        "scoring_tail_sd_multiplier": 12.0,
        "null_estimator": LRT_NULL_ESTIMATOR,
        "null_scoring_method": LRT_NULL_SCORING_METHOD,
        "null_optimizer_used": False,
        "alternative_candidate_rule": LRT_ALTERNATIVE_CANDIDATE_RULE,
        "analytic_null_loglik": analytic_null_loglik,
        "fitted_alternative_loglik": fitted_alternative_loglik,
        "analytic_null_mean": analytic_null_mean,
        "analytic_null_normal_sd": analytic_null_normal_sd,
        "analytic_null_wfcm_sigma": analytic_null_wfcm_sigma,
        "scored_candidate_selected": scored_candidate_selected,
        "statistic": float(statistic),
        "raw_statistic": float(raw_statistic),
        "embedded_null_fallback_used": raw_statistic < 0.0 if raw_defined else None,
        "materially_negative_raw": (
            raw_statistic < -negative_tolerance if raw_defined else None
        ),
        "negative_raw_magnitude": (
            max(0.0, -raw_statistic) if raw_defined else np.nan
        ),
        "negative_raw_tolerance": negative_tolerance,
        "fitting_objective_raw_statistic": fitting_objective_raw_statistic,
        "fitting_objective_nested_within_tolerance": (
            fitting_objective_raw_statistic
            >= -fitting_objective_negative_tolerance
            if fitting_raw_defined
            else None
        ),
        "fitting_objective_materially_negative": (
            fitting_objective_raw_statistic
            < -fitting_objective_negative_tolerance
            if fitting_raw_defined
            else None
        ),
        "fitting_objective_negative_tolerance": (
            fitting_objective_negative_tolerance
        ),
        "cross_objective_fallback_used": (
            raw_statistic < 0.0
            and fitting_objective_raw_statistic
            >= -fitting_objective_negative_tolerance
            if raw_defined and fitting_raw_defined
            else None
        ),
        "material_cross_objective_mismatch": (
            raw_statistic < -negative_tolerance
            and fitting_objective_raw_statistic
            >= -fitting_objective_negative_tolerance
            if raw_defined and fitting_raw_defined
            else None
        ),
        "bootstrap_pvalue": bootstrap_p,
        "chi_square_pvalue_diagnostic": chi_square_p,
        "bootstrap_reject_05": bootstrap_p <= 0.05 if np.isfinite(bootstrap_p) else None,
        "chi_square_reject_05_diagnostic": (
            chi_square_p <= 0.05 if np.isfinite(chi_square_p) else None
        ),
        "fit_success": fit_success,
        "null_fit_success": null_fit_success,
        "analysis_success": analysis_success,
        "failure_message": failure_message,
        "exception_type": exception_type,
        "gradient_inf_norm": (
            float(fit.gradient_inf_norm) if fit is not None else np.nan
        ),
        "null_gradient_inf_norm": (
            float(null_fit.gradient_inf_norm) if null_fit is not None else np.nan
        ),
        "ess_fraction": (
            float(fit.diagnostics["ess_fraction"])
            if fit is not None
            else np.nan
        ),
        "observed_runtime_seconds": observed_runtime,
        "bootstrap_runtime_seconds": bootstrap_runtime,
        "bootstrap_requested": profile.lrt_bootstrap_reps,
        "bootstrap_finite": bootstrap_finite,
        "bootstrap_valid": len(bootstrap_statistics),
        "bootstrap_optimizer_success": len(bootstrap_statistics),
        "bootstrap_negative_raw_fraction": (
            float(np.mean(boot_raw_array < 0.0))
            if bootstrap_raw_defined
            else np.nan
        ),
        "bootstrap_materially_negative_raw_fraction": (
            float(np.mean(boot_raw_array < -negative_tolerance))
            if bootstrap_raw_defined
            else np.nan
        ),
        "bootstrap_minimum_raw_statistic": (
            float(np.min(boot_raw_array)) if bootstrap_raw_defined else np.nan
        ),
        "bootstrap_fitting_objective_negative_fraction": (
            float(np.mean(boot_fitting_raw_array < 0.0))
            if bootstrap_fitting_raw_defined
            else np.nan
        ),
        "bootstrap_fitting_objective_materially_negative_fraction": float(
            np.mean(
                boot_fitting_raw_array
                < -fitting_objective_negative_tolerance
            )
        ) if bootstrap_fitting_raw_defined else np.nan,
        "bootstrap_minimum_fitting_objective_raw_statistic": (
            float(np.min(boot_fitting_raw_array))
            if bootstrap_fitting_raw_defined
            else np.nan
        ),
        "bootstrap_cross_objective_fallback_fraction": (
            float(np.mean(bootstrap_cross))
            if bootstrap_cross_defined
            else np.nan
        ),
        "bootstrap_cross_objective_fallback_count": int(
            np.sum(bootstrap_cross)
        ),
        "bootstrap_material_cross_objective_mismatch_fraction": (
            float(np.mean(bootstrap_material_cross))
            if bootstrap_cross_defined
            else np.nan
        ),
        "bootstrap_material_cross_objective_mismatch_count": int(
            np.sum(bootstrap_material_cross)
        ),
        "bootstrap_cross_objective_diagnostic_defined": int(
            boot_raw_array.size if bootstrap_cross_defined else 0
        ),
        "alternative_n_init": profile.fit_n_init,
        "null_n_init": 0,
    }


def _selection_one(profile: Profile, n: int, replicate: int, base_seed: int) -> list[dict]:
    truth = WFCMParams(
        sigma=SELECTION_TRUE_SIGMA,
        centers=np.asarray(SELECTION_TRUE_CENTERS, dtype=float)[:, None],
        weights=np.asarray(SELECTION_TRUE_WEIGHTS, dtype=float),
        m=SELECTION_FIXED_M,
    )
    data_seed = _spawn_seed(base_seed, 21, n, replicate)
    m_is = max(
        SELECTION_MINIMUM_IMPORTANCE_SIZE,
        int(np.ceil(SELECTION_IMPORTANCE_MULTIPLIER * n)),
    )
    values = None
    sampling_failure_message = ""
    sampling_exception_type = ""
    try:
        values = sample_wfcm_1d(truth, n, data_seed)
    except Exception as exc:
        sampling_failure_message = f"{type(exc).__name__}: {exc}"
        sampling_exception_type = type(exc).__name__

    candidate_seeds = {
        candidate_k: _spawn_seed(
            base_seed,
            seed_path[0],
            n,
            replicate,
            *seed_path[1:],
        )
        for candidate_k, seed_path in SELECTION_CANDIDATE_SEED_PATHS.items()
    }
    candidates: dict[int, dict] = {}
    for candidate_k in SELECTION_CANDIDATE_K:
        candidate_fit = None
        candidate_score = np.nan
        candidate_valid = False
        fit_success = False
        candidate_failure_message = sampling_failure_message
        candidate_exception_type = sampling_exception_type
        if values is not None:
            try:
                fit_kwargs = {
                    "k": candidate_k,
                    "m": truth.m,
                    "importance_size": m_is,
                    "n_init": profile.fit_n_init,
                    "maxiter": profile.maxiter,
                    "seed": candidate_seeds[candidate_k],
                }
                if candidate_k != SELECTION_TRUE_K:
                    fit_kwargs["proposal_kind"] = "defensive"
                candidate_fit = fit_wfcm(values, **fit_kwargs)
                fit_success = bool(candidate_fit.success)
                if not fit_success:
                    candidate_failure_message = (
                        str(getattr(candidate_fit, "message", ""))
                        or f"candidate k={candidate_k} did not satisfy convergence checks"
                    )
                else:
                    candidate_score = float(
                        weighted_xie_beni(values, candidate_fit.params)
                    )
                    if not np.isfinite(candidate_score):
                        raise FloatingPointError(
                            f"candidate k={candidate_k} WXBI score is nonfinite"
                        )
                    candidate_valid = True
                    candidate_failure_message = ""
                    candidate_exception_type = ""
            except Exception as exc:
                fit_success = bool(
                    candidate_fit is not None
                    and getattr(candidate_fit, "success", False)
                )
                candidate_score = np.nan
                candidate_valid = False
                candidate_failure_message = f"{type(exc).__name__}: {exc}"
                candidate_exception_type = type(exc).__name__
        candidates[candidate_k] = {
            "fit": candidate_fit,
            "fit_success": fit_success,
            "score": candidate_score,
            "valid": candidate_valid,
            "failure_message": candidate_failure_message,
            "exception_type": candidate_exception_type,
            "fit_seed": candidate_seeds[candidate_k],
        }

    # A best-among-survivors rule is not the prespecified k=2,3,4 WXBI rule.
    # Define the plug-in selection only if all three fits and scores are valid.
    selection_defined = all(
        candidates[k]["valid"] for k in SELECTION_CANDIDATE_K
    )
    selected_k = (
        min(SELECTION_CANDIDATE_K, key=lambda k: candidates[k]["score"])
        if selection_defined
        else None
    )

    missing_fisher = {
        "fisher_min_eigenvalue": np.nan,
        "fisher_max_eigenvalue": np.nan,
        "fisher_condition_clipped": np.nan,
        "fisher_eigen_floor": np.nan,
    }

    def analyze(
        analysis: str,
        record: dict | None,
        selected_truth: bool,
        fisher_seed_tag: int,
    ) -> dict:
        fit = record["fit"] if record is not None else None
        fit_success = bool(
            fit is not None and getattr(fit, "success", False)
        )
        aligned_centers = np.full_like(truth.centers, np.nan)
        lo = np.full(truth.k, np.nan)
        hi = np.full(truth.k, np.nan)
        diagnostics = dict(missing_fisher)
        inference_valid = False
        analysis_failure_message = ""
        analysis_exception_type = ""
        if selected_truth:
            if not fit_success:
                analysis_failure_message = (
                    (record or {}).get("failure_message", "")
                    or f"{analysis} fit did not satisfy convergence checks"
                )
                analysis_exception_type = (record or {}).get(
                    "exception_type", ""
                )
            else:
                try:
                    aligned_centers, _, estimated_for_truth = align_to_reference(
                        fit.params.centers, fit.params.weights, truth.centers
                    )
                    lo_raw, hi_raw, diagnostics = fisher_center_intervals(
                        fit,
                        n,
                        draws=profile.fisher_draws,
                        seed=_spawn_seed(
                            base_seed, fisher_seed_tag, n, replicate
                        ),
                    )
                    lo = np.asarray(lo_raw, dtype=float)[estimated_for_truth]
                    hi = np.asarray(hi_raw, dtype=float)[estimated_for_truth]
                    inference_valid = bool(
                        np.all(np.isfinite(aligned_centers))
                        and np.all(np.isfinite(lo))
                        and np.all(np.isfinite(hi))
                        and np.all(hi >= lo)
                    )
                    if not inference_valid:
                        raise FloatingPointError(
                            f"{analysis} Fisher interval endpoints are nonfinite or unordered"
                        )
                except Exception as exc:
                    inference_valid = False
                    lo = np.full(truth.k, np.nan)
                    hi = np.full(truth.k, np.nan)
                    diagnostics = dict(missing_fisher)
                    analysis_failure_message = f"{type(exc).__name__}: {exc}"
                    analysis_exception_type = type(exc).__name__
        fit_diagnostics = (
            getattr(fit, "diagnostics", {}) if fit is not None else {}
        )
        return {
            "analysis": analysis,
            "fit": fit,
            "fit_success": fit_success,
            "selected_truth": selected_truth,
            "inference_valid": inference_valid,
            "aligned_centers": aligned_centers,
            "lower": lo,
            "upper": hi,
            "diagnostics": diagnostics,
            "gradient_inf_norm": float(
                getattr(fit, "gradient_inf_norm", np.nan)
            ),
            "ess_fraction": float(fit_diagnostics.get("ess_fraction", np.nan)),
            "failure_message": analysis_failure_message,
            "exception_type": analysis_exception_type,
        }

    fixed_analysis = analyze(
        "fixed_k", candidates[SELECTION_TRUE_K], True, 24
    )
    selected_record = candidates[selected_k] if selection_defined else None
    if selection_defined and selected_k == SELECTION_TRUE_K:
        # Conditional post-selection coverage must condition the identical
        # k=2 inferential result on the selection event.  Recomputing Fisher
        # information with another random integration would add irrelevant
        # Monte Carlo noise and could create branch-specific failures.
        selected_analysis = {
            **fixed_analysis,
            "analysis": "selected_k_plug_in",
        }
    else:
        selected_analysis = analyze(
            "selected_k_plug_in",
            selected_record,
            False,
            25,
        )
    analyses = (fixed_analysis, selected_analysis)

    failure_parts = [
        f"candidate k={k}: {candidates[k]['failure_message']}"
        for k in SELECTION_CANDIDATE_K
        if not candidates[k]["valid"]
    ]
    failure_parts.extend(
        f"{item['analysis']}: {item['failure_message']}"
        for item in analyses
        if item["failure_message"]
        and item["failure_message"] not in failure_parts
    )
    aggregate_failure_message = "; ".join(failure_parts)
    exception_types = []
    for value in [
        *(candidates[k]["exception_type"] for k in SELECTION_CANDIDATE_K),
        *(item["exception_type"] for item in analyses),
    ]:
        if value and value not in exception_types:
            exception_types.append(value)
    aggregate_exception_type = ";".join(exception_types)

    repeated = {
        "selection_defined": selection_defined,
        "failure_message": aggregate_failure_message,
        "exception_type": aggregate_exception_type,
        **{
            f"candidate_k{k}_valid": candidates[k]["valid"]
            for k in SELECTION_CANDIDATE_K
        },
        **{
            f"candidate_k{k}_failure_message": candidates[k]["failure_message"]
            for k in SELECTION_CANDIDATE_K
        },
        **{
            f"candidate_k{k}_exception_type": candidates[k]["exception_type"]
            for k in SELECTION_CANDIDATE_K
        },
        **{
            f"candidate_k{k}_fit_seed": candidates[k]["fit_seed"]
            for k in SELECTION_CANDIDATE_K
        },
        **{
            f"xbi_k{k}": float(candidates[k]["score"])
            for k in SELECTION_CANDIDATE_K
        },
    }
    rows = []
    for item in analyses:
        analysis = item["analysis"]
        output_selected_k = (
            SELECTION_TRUE_K
            if analysis == "fixed_k"
            else selected_k if selection_defined else np.nan
        )
        for center in range(truth.k):
            conditional_cover = (
                bool(
                    item["lower"][center]
                    <= truth.centers[center, 0]
                    <= item["upper"][center]
                )
                if item["inference_valid"]
                else None
            )
            rows.append(
                {
                    "experiment": "selection",
                    "analysis": analysis,
                    "n": n,
                    "replicate": replicate,
                    "data_seed": data_seed,
                    "selected_k": output_selected_k,
                    "true_k_selected": item["selected_truth"],
                    "inference_valid": item["inference_valid"],
                    "inference_failure_message": item["failure_message"],
                    "inference_exception_type": item["exception_type"],
                    "center": center,
                    "truth": float(truth.centers[center, 0]),
                    "estimate": float(item["aligned_centers"][center, 0]),
                    "lower": float(item["lower"][center]),
                    "upper": float(item["upper"][center]),
                    "conditional_cover": conditional_cover,
                    "select_and_cover": (
                        bool(conditional_cover)
                        if item["inference_valid"]
                        else False
                    ),
                    "length": float(
                        item["upper"][center] - item["lower"][center]
                    ),
                    "fit_success": item["fit_success"],
                    "gradient_inf_norm": item["gradient_inf_norm"],
                    "ess_fraction": item["ess_fraction"],
                    **item["diagnostics"],
                    **repeated,
                }
            )
    return rows


def _flatten(chunks: Iterable[Sequence[dict] | dict]) -> list[dict]:
    rows: list[dict] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            rows.append(chunk)
        else:
            rows.extend(chunk)
    return rows


def _write_results(rows: list[dict], output: Path, stem: str) -> pd.DataFrame:
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


def summarize_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["n", "method", "center"], sort=True):
        valid = group[group["method_success"].fillna(False).astype(bool)]
        rows.append(
            {
                "n": int(keys[0]),
                "method": keys[1],
                "center": int(keys[2]),
                "attempted_replicates": int(group["replicate"].nunique()),
                "successful_replicates": int(valid["replicate"].nunique()),
                "failed_replicates": int(
                    group["replicate"].nunique() - valid["replicate"].nunique()
                ),
                "method_success_rate": float(group["method_success"].mean()),
                "coverage_successful": float(valid["covered"].mean()),
                "mean_length_successful": float(valid["length"].mean()),
                "median_length_successful": float(valid["length"].median()),
                "mean_base_runtime_seconds_all_attempts": float(
                    group["base_fit_runtime_seconds"].mean()
                ),
                "mean_method_runtime_seconds_all_attempts": float(
                    group[
                        "fisher_runtime_seconds"
                        if keys[1] == "fisher"
                        else "bootstrap_runtime_seconds"
                    ].mean()
                ),
                "mean_bootstrap_valid_all_attempts": float(
                    group["bootstrap_valid"].mean()
                ),
                "mean_ess_fraction_successful": float(
                    valid["ess_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_lrt(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, separation), group in frame.groupby(["n", "separation"], sort=True):
        valid = group[group["analysis_success"].fillna(False).astype(bool)]
        raw = pd.to_numeric(group["raw_statistic"], errors="coerce")
        raw = raw[np.isfinite(raw)]
        rows.append(
            {
                "n": int(n),
                "separation": float(separation),
                "attempted_replicates": int(group["replicate"].nunique()),
                "successful_replicates": int(valid["replicate"].nunique()),
                "failed_replicates": int(
                    group["replicate"].nunique() - valid["replicate"].nunique()
                ),
                "analysis_success_rate": float(group["analysis_success"].mean()),
                "bootstrap_rejection_rate_successful": float(
                    valid["bootstrap_reject_05"].mean()
                ),
                "chi_square_rejection_rate_diagnostic_successful": float(
                    valid["chi_square_reject_05_diagnostic"].mean()
                ),
                "mean_statistic_successful": float(valid["statistic"].mean()),
                "negative_raw_fraction_all_defined_attempts": (
                    float(np.mean(raw.to_numpy() < 0.0)) if len(raw) else np.nan
                ),
                "mean_bootstrap_finite_all_attempts": float(
                    group["bootstrap_finite"].mean()
                ),
                "mean_bootstrap_valid_all_attempts": float(
                    group["bootstrap_valid"].mean()
                ),
                "mean_end_to_end_runtime_seconds_all_attempts": float(
                    (
                        group["observed_runtime_seconds"]
                        + group["bootstrap_runtime_seconds"]
                    ).mean()
                ),
                "mean_ess_fraction_successful": float(
                    valid["ess_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = (
        frame[frame["analysis"] == "selected_k_plug_in"]
        .drop_duplicates(["n", "replicate"])
    )
    frequency_rows = []
    for n, group in selected.groupby("n", sort=True):
        defined = group[group["selection_defined"].fillna(False).astype(bool)]
        for selected_k in SELECTION_CANDIDATE_K:
            count = int((defined["selected_k"] == selected_k).sum())
            frequency_rows.append(
                {
                    "n": int(n),
                    "selected_k": selected_k,
                    "count": count,
                    "attempted_replicates": int(len(group)),
                    "defined_selections": int(len(defined)),
                    "undefined_selections": int(len(group) - len(defined)),
                    "frequency_given_complete_grid": (
                        count / len(defined) if len(defined) else np.nan
                    ),
                }
            )
    frequencies = pd.DataFrame(frequency_rows)
    coverage = (
        frame.groupby(["n", "analysis", "center"], as_index=False)
        .agg(
            true_k_selection_rate=("true_k_selected", "mean"),
            inference_valid_rate=("inference_valid", "mean"),
            conditional_coverage=("conditional_cover", "mean"),
            unconditional_select_and_cover=("select_and_cover", "mean"),
            mean_length_when_defined=("length", "mean"),
            attempted_replicates=("replicate", "nunique"),
            defined_inference_replicates=("inference_valid", "sum"),
            complete_grid_rate=("selection_defined", "mean"),
        )
    )
    return frequencies, coverage


def run_coverage(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    tasks = [(n, r) for n in profile.coverage_ns for r in range(profile.coverage_reps)]
    chunks = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_coverage_one)(profile, n, replicate, seed) for n, replicate in tasks
    )
    frame = _write_results(_flatten(chunks), output, "coverage")
    summarize_coverage(frame).to_csv(output / "coverage_summary.csv", index=False)


def run_lrt(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    separations = (0.0, 1.0, 2.0, 3.0)
    tasks = [
        (n, separation, r)
        for n in profile.lrt_ns
        for separation in separations
        for r in range(profile.lrt_reps)
    ]
    rows = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_lrt_one)(profile, n, separation, replicate, seed)
        for n, separation, replicate in tasks
    )
    frame = _write_results(rows, output, "lrt")
    summarize_lrt(frame).to_csv(output / "lrt_summary.csv", index=False)


def run_selection(profile: Profile, output: Path, jobs: int, seed: int) -> None:
    tasks = [(n, r) for n in profile.selection_ns for r in range(profile.selection_reps)]
    chunks = Parallel(n_jobs=jobs, prefer="threads", verbose=10)(
        delayed(_selection_one)(profile, n, replicate, seed) for n, replicate in tasks
    )
    frame = _write_results(_flatten(chunks), output, "selection")
    frequencies, coverage = summarize_selection(frame)
    frequencies.to_csv(output / "selection_frequency.csv", index=False)
    coverage.to_csv(output / "selection_coverage.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("coverage", "lrt", "selection", "all"))
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
                    "uncontended Fisher-versus-bootstrap runtime comparisons"
                ),
                "lrt_protocol_version": LRT_PROTOCOL_VERSION,
                "lrt_null_estimator": LRT_NULL_ESTIMATOR,
                "lrt_null_scoring_method": LRT_NULL_SCORING_METHOD,
                "lrt_alternative_candidate_rule": (
                    LRT_ALTERNATIVE_CANDIDATE_RULE
                ),
                "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
                "selection_candidate_k": list(SELECTION_CANDIDATE_K),
                "selection_true_k": SELECTION_TRUE_K,
                "selection_true_sigma": SELECTION_TRUE_SIGMA,
                "selection_true_centers": list(SELECTION_TRUE_CENTERS),
                "selection_true_weights": list(SELECTION_TRUE_WEIGHTS),
                "selection_fixed_m": SELECTION_FIXED_M,
                "selection_importance_multiplier": (
                    SELECTION_IMPORTANCE_MULTIPLIER
                ),
                "selection_minimum_importance_size": (
                    SELECTION_MINIMUM_IMPORTANCE_SIZE
                ),
                "selection_fisher_reuses_fixed_k_result": True,
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
    if args.experiment in {"coverage", "all"}:
        run_coverage(profile, args.output, args.jobs, args.seed)
    if args.experiment in {"lrt", "all"}:
        run_lrt(profile, args.output, args.jobs, args.seed)
    if args.experiment in {"selection", "all"}:
        run_selection(profile, args.output, args.jobs, args.seed)


if __name__ == "__main__":
    main()
