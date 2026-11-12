"""Numerically stable WFCM likelihood and estimation utilities.

The implementation in this module follows the manuscript parameterization

    rho(x; V, w, m)
      = [sum_j (w_j ||x-v_j||^2)^(-1/(m-1))]^(-(m-1)),

and f_theta(x) is proportional to exp{-rho(x)/sigma^2}.  In particular, the
``weights`` below are the manuscript penalty weights, not the transformed
coefficients used in the original exploratory notebook.

All optimization is carried out in identifiable reduced coordinates: log
sigma, the centers, and k-1 logits for a simplex with a positive lower bound.
The normalizer is estimated with common-random-number importance samples and
all ratios are evaluated in the log domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, linear_sum_assignment, minimize
from scipy.special import expit, gammaln, logsumexp, softmax, xlogy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


FloatArray = NDArray[np.float64]
LOG_MAX = float(np.log(np.finfo(float).max))
LOG_TINY = float(np.log(np.nextafter(0.0, 1.0)))
FIT_PROJECTED_GRADIENT_TOLERANCE = 5e-3


def _as_2d(x: ArrayLike) -> FloatArray:
    out = np.asarray(x, dtype=float)
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[0] == 0:
        raise ValueError("data must be a nonempty n-by-d array")
    if not np.all(np.isfinite(out)):
        raise ValueError("data contain nonfinite values")
    return out


def _validate_model(centers: ArrayLike, weights: ArrayLike, m: float) -> tuple[FloatArray, FloatArray]:
    v = _as_2d(centers)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.shape[0] != w.size:
        raise ValueError("there must be one weight per center")
    if not np.isfinite(m) or m <= 1.0:
        raise ValueError("m must be finite and larger than one")
    if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("all weights must be finite and strictly positive")
    if not np.isclose(w.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("weights must sum to one")
    return v, w


def squared_distances(x: ArrayLike, centers: ArrayLike) -> FloatArray:
    x_arr = _as_2d(x)
    v = _as_2d(centers)
    if x_arr.shape[1] != v.shape[1]:
        raise ValueError("data and centers have different dimensions")
    diff = x_arr[:, None, :] - v[None, :, :]
    d2 = np.einsum("nkd,nkd->nk", diff, diff, optimize=True)
    if np.any(~np.isfinite(d2)):
        raise FloatingPointError("nonfinite squared distances")
    return d2


def log_rho_and_membership(
    x: ArrayLike,
    centers: ArrayLike,
    weights: ArrayLike,
    m: float,
) -> tuple[FloatArray, FloatArray]:
    """Return log pointwise WFCM loss and geometric memberships.

    Rows that coincide exactly with one or more centers are handled by the
    directional equal-distance limit.  This convention affects a probability
    zero set under the continuous model but avoids NaNs in deterministic tests.
    """

    v, w = _validate_model(centers, weights, m)
    d2 = squared_distances(x, v)
    a = 1.0 / (m - 1.0)
    with np.errstate(divide="ignore"):
        log_terms = -a * (np.log(w)[None, :] + np.log(d2))
    log_sum = logsumexp(log_terms, axis=1)
    log_rho = -(m - 1.0) * log_sum

    u = np.empty_like(d2)
    ordinary = np.isfinite(log_sum)
    if np.any(ordinary):
        u[ordinary] = softmax(log_terms[ordinary], axis=1)
    for i in np.flatnonzero(~ordinary):
        at_center = d2[i] == 0.0
        limiting = np.where(at_center, w ** (-a), 0.0)
        u[i] = limiting / limiting.sum()

    if np.any(~np.isfinite(u)) or np.max(np.abs(u.sum(axis=1) - 1.0)) > 1e-10:
        raise FloatingPointError("membership calculation failed")
    return log_rho, u


def _safe_exp(log_x: FloatArray) -> FloatArray:
    out = np.empty_like(log_x)
    high = log_x > LOG_MAX
    low = log_x < LOG_TINY
    middle = ~(high | low)
    out[high] = np.inf
    out[low] = 0.0
    out[middle] = np.exp(log_x[middle])
    return out


def energy(
    x: ArrayLike,
    sigma: float,
    centers: ArrayLike,
    weights: ArrayLike,
    m: float,
) -> FloatArray:
    """Evaluate rho(x; V,w,m)/sigma^2 without density-scale arithmetic."""

    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    log_rho, _ = log_rho_and_membership(x, centers, weights, m)
    return _safe_exp(log_rho - 2.0 * np.log(sigma))


def membership(x: ArrayLike, centers: ArrayLike, weights: ArrayLike, m: float) -> FloatArray:
    """Geometric compatibility scores; these are not posterior probabilities."""

    return log_rho_and_membership(x, centers, weights, m)[1]


@dataclass(frozen=True)
class WFCMParams:
    sigma: float
    centers: FloatArray
    weights: FloatArray
    m: float

    def __post_init__(self) -> None:
        centers, weights = _validate_model(self.centers, self.weights, self.m)
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        centers = np.array(centers, dtype=float, copy=True)
        weights = np.array(weights, dtype=float, copy=True)
        centers.flags.writeable = False
        weights.flags.writeable = False
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "m", float(self.m))

    def validate(self) -> "WFCMParams":
        _validate_model(self.centers, self.weights, self.m)
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        return self

    @property
    def k(self) -> int:
        return int(self.centers.shape[0])

    @property
    def d(self) -> int:
        return int(self.centers.shape[1])


@dataclass(frozen=True)
class WFCMParameterization:
    """Smooth reduced parameterization with weights bounded below."""

    k: int
    d: int
    m_fixed: float | None = 2.0
    m_bounds: tuple[float, float] = (1.15, 4.0)
    weight_floor: float = 1e-4

    def __post_init__(self) -> None:
        if (
            isinstance(self.k, (bool, np.bool_))
            or isinstance(self.d, (bool, np.bool_))
            or not isinstance(self.k, (int, np.integer))
            or not isinstance(self.d, (int, np.integer))
            or self.k < 1
            or self.d < 1
        ):
            raise ValueError("k and d must be positive integers")
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "d", int(self.d))
        if not (0.0 <= self.weight_floor < 1.0 / self.k):
            raise ValueError("weight_floor must be in [0,1/k)")
        if self.m_fixed is not None and (not np.isfinite(self.m_fixed) or self.m_fixed <= 1.0):
            raise ValueError("m_fixed must exceed one")
        if (
            not np.all(np.isfinite(self.m_bounds))
            or self.m_bounds[0] <= 1.0
            or self.m_bounds[0] >= self.m_bounds[1]
        ):
            raise ValueError("invalid m bounds")

    @property
    def size(self) -> int:
        return 1 + self.k * self.d + (self.k - 1) + (self.m_fixed is None)

    @property
    def center_slice(self) -> slice:
        return slice(1, 1 + self.k * self.d)

    @property
    def logit_slice(self) -> slice:
        start = 1 + self.k * self.d
        return slice(start, start + self.k - 1)

    @property
    def m_index(self) -> int | None:
        return None if self.m_fixed is not None else self.size - 1

    def _weights_from_logits(self, logits: FloatArray) -> tuple[FloatArray, FloatArray]:
        augmented = np.concatenate([np.asarray(logits, dtype=float), [0.0]])
        p = softmax(augmented)
        scale = 1.0 - self.k * self.weight_floor
        return self.weight_floor + scale * p, p

    def unpack_with_aux(self, z: ArrayLike) -> tuple[WFCMParams, FloatArray, float]:
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        if z_arr.size != self.size or np.any(~np.isfinite(z_arr)):
            raise ValueError(f"expected {self.size} finite free parameters")
        sigma = float(np.exp(z_arr[0]))
        centers = z_arr[self.center_slice].reshape(self.k, self.d)
        weights, p = self._weights_from_logits(z_arr[self.logit_slice])
        if self.m_fixed is None:
            lo, hi = self.m_bounds
            s = float(expit(z_arr[self.m_index]))
            m = lo + (hi - lo) * s
            dm_dxi = (hi - lo) * s * (1.0 - s)
        else:
            m = float(self.m_fixed)
            dm_dxi = 0.0
        return WFCMParams(sigma, centers, weights, m), p, dm_dxi

    def unpack(self, z: ArrayLike) -> WFCMParams:
        return self.unpack_with_aux(z)[0]

    def pack(self, params: WFCMParams) -> FloatArray:
        params.validate()
        if params.k != self.k or params.d != self.d:
            raise ValueError("parameter dimensions do not match specification")
        if self.m_fixed is not None and not np.isclose(
            params.m, self.m_fixed, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("parameter m does not match fixed m")
        if np.any(params.weights <= self.weight_floor):
            raise ValueError("weights must be strictly above the parameterization floor")
        scale = 1.0 - self.k * self.weight_floor
        p = (params.weights - self.weight_floor) / scale
        p = np.clip(p, 1e-12, None)
        p /= p.sum()
        logits = np.log(p[:-1]) - np.log(p[-1])
        pieces = [[np.log(params.sigma)], params.centers.reshape(-1), logits]
        if self.m_fixed is None:
            lo, hi = self.m_bounds
            if not lo < params.m < hi:
                raise ValueError("parameter m must lie strictly inside m_bounds")
            s = (params.m - lo) / (hi - lo)
            pieces.append([np.log(s / (1.0 - s))])
        return np.concatenate([np.asarray(piece, dtype=float).reshape(-1) for piece in pieces])

    def bounds(self, x: ArrayLike) -> list[tuple[float, float]]:
        data = _as_2d(x)
        spread = np.ptp(data, axis=0)
        fallback = np.std(data, axis=0)
        spread = np.where(spread > 1e-8, spread, np.maximum(fallback, 1.0))
        lower = np.min(data, axis=0) - 0.75 * spread
        upper = np.max(data, axis=0) + 0.75 * spread
        scale = float(np.sqrt(np.mean(np.var(data, axis=0))) + 1e-8)
        out: list[tuple[float, float]] = [(np.log(scale * 1e-3), np.log(scale * 1e3))]
        for _ in range(self.k):
            out.extend(zip(lower.tolist(), upper.tolist(), strict=True))
        out.extend([(-18.0, 18.0)] * (self.k - 1))
        if self.m_fixed is None:
            out.append((-14.0, 14.0))
        return out


class Proposal(Protocol):
    name: str

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray: ...

    def logpdf(self, x: ArrayLike) -> FloatArray: ...


@dataclass
class GaussianMixtureProposal:
    model: GaussianMixture
    name: str

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:
        if size < 1:
            raise ValueError("proposal sample size must be positive")
        # sklearn's sample uses model.random_state.  Resetting it from our
        # generator makes every context seed explicit and reproducible.
        self.model.random_state = int(rng.integers(0, 2**31 - 1))
        values, _ = self.model.sample(size)
        return np.asarray(values, dtype=float)

    def logpdf(self, x: ArrayLike) -> FloatArray:
        return np.asarray(self.model.score_samples(_as_2d(x)), dtype=float)


@dataclass
class DefensiveMixtureProposal:
    base: GaussianMixtureProposal
    location: FloatArray
    scale: FloatArray
    df: float = 5.0
    defensive_fraction: float = 0.1
    name: str = "defensive_gmm_t"
    _chol: FloatArray = field(init=False, repr=False)
    _logdet: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.defensive_fraction < 1.0:
            raise ValueError("defensive_fraction must lie in (0,1)")
        if not np.isfinite(self.df) or self.df <= 0.0:
            raise ValueError("Student-t degrees of freedom must be positive")
        self.location = np.asarray(self.location, dtype=float).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=float)
        d = self.location.size
        if (
            d == 0
            or self.scale.shape != (d, d)
            or np.any(~np.isfinite(self.location))
            or np.any(~np.isfinite(self.scale))
            or not np.allclose(self.scale, self.scale.T, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("Student-t location and scale have incompatible values")
        try:
            self._chol = np.linalg.cholesky(self.scale)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Student-t scale must be positive definite")
        self._logdet = float(2.0 * np.sum(np.log(np.diag(self._chol))))

    def _t_logpdf(self, x: ArrayLike) -> FloatArray:
        values = _as_2d(x)
        if values.shape[1] != self.location.size:
            raise ValueError("data and Student-t proposal have different dimensions")
        centered = values - self.location[None, :]
        whitened = np.linalg.solve(self._chol, centered.T).T
        mahal = np.einsum("nd,nd->n", whitened, whitened, optimize=True)
        d = values.shape[1]
        constant = (
            gammaln((self.df + d) / 2.0)
            - gammaln(self.df / 2.0)
            - 0.5 * (d * np.log(self.df * np.pi) + self._logdet)
        )
        return constant - 0.5 * (self.df + d) * np.log1p(mahal / self.df)

    def sample(self, size: int, rng: np.random.Generator) -> FloatArray:
        if size < 1:
            raise ValueError("proposal sample size must be positive")
        choose_t = rng.random(size) < self.defensive_fraction
        out = np.empty((size, self.location.size), dtype=float)
        n_t = int(choose_t.sum())
        if n_t:
            gaussian = rng.multivariate_normal(np.zeros(self.location.size), self.scale, size=n_t)
            chi = rng.chisquare(self.df, size=n_t)
            out[choose_t] = self.location + gaussian / np.sqrt(chi / self.df)[:, None]
        if n_t < size:
            out[~choose_t] = self.base.sample(size - n_t, rng)
        return out

    def logpdf(self, x: ArrayLike) -> FloatArray:
        values = _as_2d(x)
        return logsumexp(
            np.vstack(
                [
                    np.log1p(-self.defensive_fraction) + self.base.logpdf(values),
                    np.log(self.defensive_fraction) + self._t_logpdf(values),
                ]
            ),
            axis=0,
        )


def build_proposal(
    x: ArrayLike,
    k: int,
    kind: str = "defensive",
    seed: int = 0,
) -> Proposal:
    data = _as_2d(x)
    if kind not in {"defensive", "gmm_full", "gmm_diag"}:
        raise ValueError(f"unknown proposal kind: {kind}")
    if not isinstance(k, (int, np.integer)) or k < 1 or k > data.shape[0]:
        raise ValueError("proposal component count must be between one and n")
    covariance_type = "diag" if kind == "gmm_diag" else "full"
    model = GaussianMixture(
        n_components=int(k),
        covariance_type=covariance_type,
        reg_covar=max(1e-6, 1e-6 * float(np.mean(np.var(data, axis=0)) + 1.0)),
        n_init=2,
        max_iter=300,
        random_state=seed,
    ).fit(data)
    base = GaussianMixtureProposal(model=model, name=f"gmm_{covariance_type}")
    if kind in {"gmm_full", "gmm_diag"}:
        return base
    cov = np.atleast_2d(np.cov(data, rowvar=False))
    if cov.shape != (data.shape[1], data.shape[1]) or np.any(~np.isfinite(cov)):
        cov = np.eye(data.shape[1]) * float(np.var(data))
    if np.any(~np.isfinite(cov)):
        cov = np.eye(data.shape[1])
    ridge = max(1e-6, 1e-6 * float(np.trace(cov) / data.shape[1] + 1.0))
    scale = 4.0 * (cov + ridge * np.eye(data.shape[1]))
    return DefensiveMixtureProposal(base, np.mean(data, axis=0), scale)


@dataclass(frozen=True)
class ISContext:
    samples: FloatArray
    log_q: FloatArray
    proposal_name: str
    seed: int

    def __post_init__(self) -> None:
        samples = np.array(_as_2d(self.samples), dtype=float, copy=True)
        log_q = np.array(self.log_q, dtype=float, copy=True).reshape(-1)
        if log_q.size != samples.shape[0] or np.any(~np.isfinite(log_q)):
            raise ValueError("importance samples and log densities have incompatible values")
        if not self.proposal_name:
            raise ValueError("proposal name must be nonempty")
        samples.flags.writeable = False
        log_q.flags.writeable = False
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_q", log_q)
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def draw(cls, proposal: Proposal, size: int, seed: int) -> "ISContext":
        if size < 2:
            raise ValueError("importance sample size must be at least two")
        rng = np.random.default_rng(seed)
        samples = proposal.sample(size, rng)
        log_q = proposal.logpdf(samples)
        if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(log_q)):
            raise FloatingPointError("proposal produced nonfinite samples or log densities")
        return cls(samples, log_q, proposal.name, seed)

    @property
    def size(self) -> int:
        return int(self.samples.shape[0])


def log_normalizer(params: WFCMParams, context: ISContext) -> tuple[float, FloatArray]:
    e = energy(context.samples, params.sigma, params.centers, params.weights, params.m)
    log_iw = -e - context.log_q
    if np.any(~np.isfinite(log_iw)):
        raise FloatingPointError("nonfinite importance log weights")
    value = float(logsumexp(log_iw) - np.log(context.size))
    normalized = softmax(log_iw)
    return value, normalized


def importance_diagnostics(params: WFCMParams, context: ISContext) -> dict[str, float]:
    log_z, normalized = log_normalizer(params, context)
    ess = float(1.0 / np.sum(normalized**2))
    return {
        "log_normalizer": log_z,
        "ess": ess,
        "ess_fraction": ess / context.size,
        "max_normalized_weight": float(np.max(normalized)),
    }


def energy_and_gradient_free(
    z: ArrayLike,
    x: ArrayLike,
    spec: WFCMParameterization,
) -> tuple[FloatArray, FloatArray]:
    """Energy and its row-wise gradient in the reduced coordinates."""

    data = _as_2d(x)
    params, p, dm_dxi = spec.unpack_with_aux(z)
    d2 = squared_distances(data, params.centers)
    log_rho, u = log_rho_and_membership(data, params.centers, params.weights, params.m)
    rho = _safe_exp(log_rho)
    e = rho / params.sigma**2
    n = data.shape[0]
    grad = np.empty((n, spec.size), dtype=float)
    grad[:, 0] = -2.0 * e

    # Envelope theorem for rho = min_u sum_j w_j u_j^m ||x-v_j||^2.
    u_m = u**params.m
    center_grad = (
        2.0
        * params.weights[None, :, None]
        * u_m[:, :, None]
        * (params.centers[None, :, :] - data[:, None, :])
        / params.sigma**2
    )
    grad[:, spec.center_slice] = center_grad.reshape(n, spec.k * spec.d)

    if spec.k > 1:
        g_w = u_m * d2 / params.sigma**2
        scale = 1.0 - spec.k * spec.weight_floor
        p_average = g_w @ p
        g_eta = scale * p[None, :-1] * (g_w[:, :-1] - p_average[:, None])
        grad[:, spec.logit_slice] = g_eta

    if spec.m_fixed is None:
        # xlogy(u^m,u) gives zero at u=0 without creating 0 * log(0).
        d_rho_dm = np.sum(params.weights[None, :] * d2 * xlogy(u_m, u), axis=1)
        grad[:, spec.m_index] = (d_rho_dm / params.sigma**2) * dm_dxi

    if np.any(~np.isfinite(e)) or np.any(~np.isfinite(grad)):
        raise FloatingPointError("nonfinite energy or derivative")
    return e, grad


def average_nll_and_gradient(
    z: ArrayLike,
    x: ArrayLike,
    context: ISContext,
    spec: WFCMParameterization,
) -> tuple[float, FloatArray]:
    e_data, g_data = energy_and_gradient_free(z, x, spec)
    e_is, g_is = energy_and_gradient_free(z, context.samples, spec)
    log_iw = -e_is - context.log_q
    if np.any(~np.isfinite(log_iw)):
        return np.inf, np.full(spec.size, np.nan)
    alpha = softmax(log_iw)
    value = float(np.mean(e_data) + logsumexp(log_iw) - np.log(context.size))
    gradient = np.mean(g_data, axis=0) - alpha @ g_is
    if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
        return np.inf, np.full(spec.size, np.nan)
    return value, np.asarray(gradient, dtype=float)


@dataclass
class FitResult:
    params: WFCMParams
    free_params: FloatArray
    average_nll: float
    success: bool
    optimizer_success: bool
    message: str
    iterations: int
    function_evaluations: int
    gradient_inf_norm: float
    runtime_seconds: float
    diagnostics: dict[str, float]
    proposal_name: str
    importance_seed: int
    fit_seed: int
    parameterization: WFCMParameterization | None = field(default=None, repr=False)
    importance_context: ISContext | None = field(default=None, repr=False)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-ready record without the potentially large IS draws."""

        return {
            "params": {
                "sigma": self.params.sigma,
                "centers": self.params.centers.tolist(),
                "weights": self.params.weights.tolist(),
                "m": self.params.m,
            },
            "free_params": self.free_params.tolist(),
            "average_nll": self.average_nll,
            "success": self.success,
            "optimizer_success": self.optimizer_success,
            "message": self.message,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "gradient_inf_norm": self.gradient_inf_norm,
            "runtime_seconds": self.runtime_seconds,
            "diagnostics": dict(self.diagnostics),
            "proposal_name": self.proposal_name,
            "importance_seed": self.importance_seed,
            "fit_seed": self.fit_seed,
        }


def _initial_params(data: FloatArray, k: int, m: float, seed: int) -> WFCMParams:
    kmeans = KMeans(n_clusters=k, n_init=1, init="k-means++", random_state=seed).fit(data)
    centers = np.asarray(kmeans.cluster_centers_, dtype=float)
    weights = np.full(k, 1.0 / k)
    log_rho, _ = log_rho_and_membership(data, centers, weights, m)
    rho = _safe_exp(log_rho)
    sigma = float(np.sqrt(max(2.0 * np.mean(rho) / data.shape[1], 1e-8)))
    return WFCMParams(sigma, centers, weights, m)


def optimize_with_context(
    x: ArrayLike,
    context: ISContext,
    spec: WFCMParameterization,
    initial: WFCMParams,
    fit_seed: int,
    maxiter: int = 400,
    gtol: float = 2e-5,
) -> tuple[OptimizeResult, float]:
    data = _as_2d(x)
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if not np.isfinite(gtol) or gtol <= 0.0:
        raise ValueError("gtol must be finite and positive")
    z0 = spec.pack(initial)

    def objective(z: FloatArray) -> tuple[float, FloatArray]:
        try:
            return average_nll_and_gradient(z, data, context, spec)
        except (FloatingPointError, ValueError, OverflowError):
            return np.inf, np.zeros_like(z)

    start = perf_counter()
    result = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        jac=True,
        bounds=spec.bounds(data),
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": gtol, "maxls": 40},
    )
    return result, perf_counter() - start


def _projected_gradient_inf_norm(
    point: ArrayLike,
    gradient: ArrayLike,
    bounds: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    """Return projected and raw infinity norms for box-constrained minimization."""

    z = np.asarray(point, dtype=float).reshape(-1)
    raw = np.asarray(gradient, dtype=float).reshape(-1)
    if z.size != raw.size or len(bounds) != z.size or np.any(~np.isfinite(raw)):
        return np.inf, np.inf
    projected = raw.copy()
    for index, (lower, upper) in enumerate(bounds):
        lower_tolerance = 1e-8 * (1.0 + abs(lower))
        upper_tolerance = 1e-8 * (1.0 + abs(upper))
        # At a lower bound, a positive derivative points downhill outside the
        # feasible box; the analogous condition at an upper bound is negative.
        if z[index] <= lower + lower_tolerance and raw[index] > 0.0:
            projected[index] = 0.0
        elif z[index] >= upper - upper_tolerance and raw[index] < 0.0:
            projected[index] = 0.0
    return float(np.max(np.abs(projected), initial=0.0)), float(
        np.max(np.abs(raw), initial=0.0)
    )


def fit_wfcm(
    x: ArrayLike,
    k: int,
    m: float = 2.0,
    importance_size: int = 5_000,
    proposal_kind: str = "defensive",
    n_init: int = 3,
    maxiter: int = 400,
    seed: int = 0,
    weight_floor: float = 1e-4,
    initial_params: WFCMParams | None = None,
    context: ISContext | None = None,
) -> FitResult:
    """Fit the finite-IS approximate MLE with common random numbers."""

    overall_start = perf_counter()
    data = _as_2d(x)
    if (
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
        or k > data.shape[0]
    ):
        raise ValueError("invalid number of clusters")
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if context is None and (
        not isinstance(importance_size, (int, np.integer)) or importance_size < 2
    ):
        raise ValueError("importance_size must be an integer of at least two")
    spec = WFCMParameterization(k, data.shape[1], m_fixed=m, weight_floor=weight_floor)
    rng = np.random.default_rng(seed)
    if context is None:
        proposal = build_proposal(data, k, proposal_kind, int(rng.integers(0, 2**31 - 1)))
        is_seed = int(rng.integers(0, 2**31 - 1))
        context = ISContext.draw(proposal, importance_size, is_seed)
    else:
        is_seed = context.seed
        if context.samples.shape[1] != data.shape[1]:
            raise ValueError("importance context and data have different dimensions")

    if initial_params is not None:
        initial_params.validate()
        if initial_params.k != k or initial_params.d != data.shape[1]:
            raise ValueError("initial parameters have incompatible dimensions")
        if not np.isclose(initial_params.m, m, rtol=1e-12, atol=1e-12):
            raise ValueError("initial parameters use a different m")

    candidates: list[tuple[OptimizeResult, float]] = []
    starts: list[WFCMParams] = []
    if initial_params is not None:
        starts.append(initial_params)
    while len(starts) < n_init:
        starts.append(_initial_params(data, k, m, int(rng.integers(0, 2**31 - 1))))

    optimizer_time = 0.0
    for idx, initial in enumerate(starts[:n_init]):
        result, elapsed = optimize_with_context(
            data,
            context,
            spec,
            initial,
            fit_seed=seed + idx,
            maxiter=maxiter,
        )
        optimizer_time += elapsed
        if np.isfinite(result.fun):
            candidates.append((result, elapsed))
    if not candidates:
        raise RuntimeError("all WFCM optimization starts failed")
    best, _ = min(candidates, key=lambda pair: float(pair[0].fun))
    params = spec.unpack(best.x)
    diagnostics = importance_diagnostics(params, context)
    grad_norm, raw_grad_norm = _projected_gradient_inf_norm(
        best.x, best.jac, spec.bounds(data)
    )
    success = bool(
        best.success
        and np.isfinite(best.fun)
        and np.all(np.isfinite(best.x))
        and grad_norm <= FIT_PROJECTED_GRADIENT_TOLERANCE
    )
    diagnostics["optimizer_runtime_seconds"] = optimizer_time
    diagnostics["raw_gradient_inf_norm"] = raw_grad_norm
    return FitResult(
        params=params,
        free_params=np.asarray(best.x, dtype=float),
        average_nll=float(best.fun),
        success=success,
        optimizer_success=bool(best.success),
        message=str(best.message),
        iterations=int(best.nit),
        function_evaluations=int(best.nfev),
        gradient_inf_norm=grad_norm,
        runtime_seconds=perf_counter() - overall_start,
        diagnostics=diagnostics,
        proposal_name=context.proposal_name,
        importance_seed=is_seed,
        fit_seed=seed,
        parameterization=spec,
        importance_context=context,
    )


def model_fisher_information(
    fit: FitResult,
    context: ISContext | None = None,
    spec: WFCMParameterization | None = None,
    ridge: float = 1e-8,
) -> tuple[FloatArray, FloatArray]:
    """Self-normalized IS estimate of I(theta)=Var_theta[d E_theta(X)]."""

    context = context if context is not None else fit.importance_context
    spec = spec if spec is not None else fit.parameterization
    if context is None or spec is None:
        raise ValueError("the fit does not contain its IS context and parameterization")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if context.samples.shape[1] != fit.params.d or spec.k != fit.params.k or spec.d != fit.params.d:
        raise ValueError("fit, context, and parameterization are incompatible")
    represented = spec.unpack(fit.free_params)
    if (
        not np.isclose(represented.sigma, fit.params.sigma)
        or not np.allclose(represented.centers, fit.params.centers)
        or not np.allclose(represented.weights, fit.params.weights)
        or not np.isclose(represented.m, fit.params.m)
    ):
        raise ValueError("parameterization does not represent the supplied fit")
    e, gradients = energy_and_gradient_free(fit.free_params, context.samples, spec)
    alpha = softmax(-e - context.log_q)
    mean = alpha @ gradients
    centered = gradients - mean
    information = (centered * alpha[:, None]).T @ centered
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    floor = max(ridge, ridge * float(np.max(eigenvalues, initial=1.0)))
    stabilized = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    inverse = (eigenvectors * (1.0 / np.maximum(eigenvalues, floor))) @ eigenvectors.T
    return stabilized, inverse


def empirical_score_information(
    fit: FitResult,
    x: ArrayLike,
    context: ISContext | None = None,
    spec: WFCMParameterization | None = None,
) -> FloatArray:
    context = context if context is not None else fit.importance_context
    spec = spec if spec is not None else fit.parameterization
    if context is None or spec is None:
        raise ValueError("the fit does not contain its IS context and parameterization")
    if context.samples.shape[1] != fit.params.d or spec.k != fit.params.k or spec.d != fit.params.d:
        raise ValueError("fit, context, and parameterization are incompatible")
    represented = spec.unpack(fit.free_params)
    if (
        not np.isclose(represented.sigma, fit.params.sigma)
        or not np.allclose(represented.centers, fit.params.centers)
        or not np.allclose(represented.weights, fit.params.weights)
        or not np.isclose(represented.m, fit.params.m)
    ):
        raise ValueError("parameterization does not represent the supplied fit")
    e_is, g_is = energy_and_gradient_free(fit.free_params, context.samples, spec)
    mean_g = softmax(-e_is - context.log_q) @ g_is
    _, g_data = energy_and_gradient_free(fit.free_params, x, spec)
    scores = -g_data + mean_g
    return scores.T @ scores / scores.shape[0]


def align_to_reference(
    centers: ArrayLike,
    weights: ArrayLike,
    reference_centers: ArrayLike,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Return estimates in reference-label order using optimal assignment."""

    estimated = _as_2d(centers)
    reference = _as_2d(reference_centers)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if (
        estimated.shape != reference.shape
        or w.size != estimated.shape[0]
        or np.any(~np.isfinite(w))
    ):
        raise ValueError("alignment shapes do not match")
    cost = np.linalg.norm(estimated[:, None, :] - reference[None, :, :], axis=2)
    rows, columns = linear_sum_assignment(cost)
    aligned_centers = np.empty_like(reference)
    aligned_weights = np.empty_like(w)
    estimated_index_for_reference = np.empty(w.size, dtype=np.int64)
    for estimated_index, reference_index in zip(rows, columns, strict=True):
        aligned_centers[reference_index] = estimated[estimated_index]
        aligned_weights[reference_index] = w[estimated_index]
        estimated_index_for_reference[reference_index] = estimated_index
    return aligned_centers, aligned_weights, estimated_index_for_reference


def weighted_xie_beni(
    x: ArrayLike,
    params: WFCMParams,
) -> float:
    """Weighted Xie--Beni heuristic using relative weights ``k*w_j``.

    The factor ``k`` is essential when scores are compared across candidate
    cluster counts: it removes the arbitrary ``1/k`` scaling caused by the
    simplex convention and makes equal-weight WFCM exactly recover classical
    Xie--Beni.  This remains a cluster-validity heuristic, not a likelihood
    model-selection criterion.
    """

    data = _as_2d(x)
    if params.k < 2:
        return np.inf
    u = membership(data, params.centers, params.weights, params.m)
    d2 = squared_distances(data, params.centers)
    # The model weights are simplex-normalized.  Their direct use would make
    # the balanced case smaller by exactly 1/k, mechanically favouring larger
    # candidate k.  Relative weights k*w_j retain the intended reweighting and
    # reduce exactly to the classical Xie--Beni numerator when w_j=1/k.
    relative_weights = params.k * params.weights
    numerator = float(np.sum(relative_weights[None, :] * u**params.m * d2))
    center_d2 = squared_distances(params.centers, params.centers)
    np.fill_diagonal(center_d2, np.inf)
    separation = float(np.min(center_d2))
    if not np.isfinite(separation) or separation <= 0.0:
        return np.inf
    return numerator / (data.shape[0] * separation)


def select_m_grid(
    x: ArrayLike,
    k: int,
    grid: Sequence[float],
    importance_size: int = 5_000,
    proposal_kind: str = "defensive",
    n_init: int = 2,
    seed: int = 0,
) -> tuple[FitResult, list[FitResult]]:
    data = _as_2d(x)
    values = [float(value) for value in grid]
    if not values or any(not np.isfinite(value) or value <= 1.0 for value in values):
        raise ValueError("m grid must contain values larger than one")
    proposal = build_proposal(data, k, proposal_kind, seed + 1)
    context = ISContext.draw(proposal, importance_size, seed + 2)
    fits = [
        fit_wfcm(
            data,
            k,
            m=value,
            importance_size=importance_size,
            proposal_kind=proposal_kind,
            n_init=n_init,
            seed=seed + 1009 * index,
            context=context,
        )
        for index, value in enumerate(values)
    ]
    successful = [fit for fit in fits if fit.success]
    if not successful:
        raise RuntimeError("all m-grid fits failed the convergence checks")
    return min(successful, key=lambda fit: fit.average_nll), fits


def select_k_weighted_xb(
    x: ArrayLike,
    candidates: Sequence[int],
    m: float = 2.0,
    importance_size: int = 3_000,
    proposal_kind: str = "defensive",
    n_init: int = 2,
    seed: int = 0,
) -> tuple[FitResult, list[tuple[int, float, FitResult]]]:
    data = _as_2d(x)
    raw_values = list(candidates)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw_values
    ):
        raise ValueError("candidate k values must be integers")
    values = [int(value) for value in raw_values]
    if (
        not values
        or len(set(values)) != len(values)
        or any(value < 2 or value > data.shape[0] for value in values)
    ):
        raise ValueError("candidate k values must be unique integers between two and n")
    records: list[tuple[int, float, FitResult]] = []
    for index, k in enumerate(values):
        fit = fit_wfcm(
            data,
            int(k),
            m=m,
            importance_size=importance_size,
            proposal_kind=proposal_kind,
            n_init=n_init,
            seed=seed + 1009 * index,
        )
        score = weighted_xie_beni(data, fit.params) if fit.success else np.inf
        records.append((int(k), score, fit))
    eligible = [record for record in records if np.isfinite(record[1])]
    if not eligible:
        raise RuntimeError("all candidate-k fits failed the convergence checks")
    selected = min(eligible, key=lambda record: record[1])[2]
    return selected, records


def sample_wfcm_1d(
    params: WFCMParams,
    size: int,
    seed: int,
    grid_size: int = 200_001,
    tail_sd: float = 11.0,
) -> FloatArray:
    """Numerically invert a high-resolution one-dimensional WFCM CDF.

    This produces independent draws and is used for inferential simulations,
    avoiding the serial dependence of the exploratory notebook's single-chain
    Metropolis-Hastings samples.
    """

    params.validate()
    if params.d != 1:
        raise ValueError("sample_wfcm_1d is only defined for d=1")
    if (
        not isinstance(size, (int, np.integer))
        or not isinstance(grid_size, (int, np.integer))
        or size < 1
        or grid_size < 10_001
        or not np.isfinite(tail_sd)
        or tail_sd <= 0.0
    ):
        raise ValueError("invalid sample or grid size")
    a = 1.0 / (params.m - 1.0)
    tail_coefficient = (
        np.sum(params.weights ** (-a)) ** (-(params.m - 1.0)) / params.sigma**2
    )
    asymptotic_sd = np.sqrt(1.0 / (2.0 * tail_coefficient))
    centers = params.centers[:, 0]
    lo = float(np.min(centers) - tail_sd * asymptotic_sd - np.ptp(centers))
    hi = float(np.max(centers) + tail_sd * asymptotic_sd + np.ptp(centers))
    grid = np.linspace(lo, hi, grid_size)
    log_unnormalized = -energy(
        grid[:, None], params.sigma, params.centers, params.weights, params.m
    )
    density_values = np.exp(log_unnormalized - np.max(log_unnormalized))
    increments = 0.5 * (density_values[1:] + density_values[:-1]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    cdf /= cdf[-1]
    if not np.isclose(cdf[-1], 1.0) or np.any(np.diff(cdf) < 0.0):
        raise FloatingPointError("failed to construct a valid numerical CDF")
    rng = np.random.default_rng(seed)
    return np.interp(rng.random(size), cdf, grid)[:, None]
