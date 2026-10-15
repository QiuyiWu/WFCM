"""Reproducible labelled scenarios for the clustering benchmark.

The Gaussian scenarios cross a directly recorded imbalance ratio with a
center-separation control.  ``irregular`` uses two interlocking moons and
crosses the same imbalance ratio with the moon noise level.  Every generator
uses exact (rather than random) cluster counts so that a factorial comparison
does not accidentally confound imbalance with Monte Carlo count variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

IMBALANCE_RATIOS: dict[str, float] = {
    "balanced": 1.0,
    "moderate": 4.0,
    "severe": 10.0,
}

# Center separation in units of the spherical within-cluster standard
# deviation.  Larger values imply less overlap.
GAUSSIAN_SEPARATIONS: dict[str, float] = {
    "low": 4.0,
    "moderate": 3.0,
    "medium": 3.0,
    "high": 2.0,
}

# Standard deviation used by sklearn's two-moons generator.  Larger values
# imply more overlap/noisier cluster boundaries.
IRREGULAR_NOISE: dict[str, float] = {
    "low": 0.05,
    "moderate": 0.12,
    "medium": 0.12,
    "high": 0.20,
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@dataclass(frozen=True)
class ScenarioData:
    x: FloatArray
    labels: IntArray
    true_centers: FloatArray
    scenario: str
    imbalance_ratio: float
    overlap: str | float
    seed: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        centers = np.asarray(self.true_centers, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or np.any(~np.isfinite(x)):
            raise ValueError("x must be a finite n-by-d array with n at least two")
        if labels.shape != (x.shape[0],) or np.any(labels < 0):
            raise ValueError("labels must contain one nonnegative label per observation")
        observed = np.unique(labels)
        if not np.array_equal(observed, np.arange(observed.size)):
            raise ValueError("labels must be contiguous and start at zero")
        if centers.shape != (observed.size, x.shape[1]) or np.any(~np.isfinite(centers)):
            raise ValueError("true_centers must contain one finite center per label")
        if not np.isfinite(self.imbalance_ratio) or self.imbalance_ratio < 1.0:
            raise ValueError("imbalance_ratio must be finite and at least one")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "true_centers", centers)
        object.__setattr__(self, "scenario", str(self.scenario))
        object.__setattr__(self, "imbalance_ratio", float(self.imbalance_ratio))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def d(self) -> int:
        return int(self.x.shape[1])

    @property
    def k(self) -> int:
        return int(self.true_centers.shape[0])

    @property
    def counts(self) -> IntArray:
        return np.bincount(self.labels, minlength=self.k).astype(np.int64)

    def to_record(self, include_data: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "scenario": self.scenario,
            "n": self.n,
            "d": self.d,
            "k": self.k,
            "imbalance_ratio": self.imbalance_ratio,
            "overlap": self.overlap,
            "seed": self.seed,
            "counts": self.counts.tolist(),
            "true_centers": self.true_centers.tolist(),
            "metadata": _json_ready(self.metadata),
        }
        if include_data:
            record["x"] = self.x.tolist()
            record["labels"] = self.labels.tolist()
        return record


def _resolve_imbalance(value: str | float) -> tuple[float, str | float]:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in IMBALANCE_RATIOS:
            raise ValueError(f"unknown imbalance level: {value}")
        return IMBALANCE_RATIOS[key], key
    ratio = float(value)
    if not np.isfinite(ratio) or ratio < 1.0:
        raise ValueError("numeric imbalance must be finite and at least one")
    return ratio, ratio


def _exact_counts(n: int, k: int, ratio: float) -> IntArray:
    if not isinstance(n, (int, np.integer)) or not isinstance(k, (int, np.integer)):
        raise ValueError("n and k must be integers")
    if k < 2 or n < k:
        raise ValueError("require two or more clusters and at least one observation per cluster")
    probabilities = np.r_[ratio, np.ones(k - 1)]
    probabilities /= probabilities.sum()
    raw = n * probabilities
    counts = np.floor(raw).astype(np.int64)
    counts[counts < 1] = 1

    while int(counts.sum()) < n:
        priorities = raw - counts
        counts[int(np.argmax(priorities))] += 1
    while int(counts.sum()) > n:
        eligible = counts > 1
        if not np.any(eligible):  # pragma: no cover - n >= k guarantees feasibility
            raise RuntimeError("could not construct positive cluster counts")
        surplus = counts - raw
        surplus[~eligible] = -np.inf
        counts[int(np.argmax(surplus))] -= 1
    return counts


def _resolve_overlap(
    value: str | float, mapping: dict[str, float], *, allow_zero: bool
) -> tuple[float, str | float]:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in mapping:
            raise ValueError(f"unknown overlap level: {value}")
        return mapping[key], key
    numeric = float(value)
    valid = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not np.isfinite(numeric) or not valid:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"numeric overlap control must be finite and {qualifier}")
    return numeric, numeric


def _circle_centers(k: int, d: int, adjacent_separation: float) -> FloatArray:
    if d == 1:
        return (np.arange(k, dtype=float) - 0.5 * (k - 1))[:, None] * adjacent_separation
    radius = adjacent_separation / (2.0 * np.sin(np.pi / k))
    angles = 2.0 * np.pi * np.arange(k) / k
    centers = np.zeros((k, d), dtype=float)
    centers[:, 0] = radius * np.cos(angles)
    centers[:, 1] = radius * np.sin(angles)
    return centers


def _shuffle(x: FloatArray, labels: IntArray, rng: np.random.Generator) -> tuple[FloatArray, IntArray]:
    order = rng.permutation(x.shape[0])
    return x[order], labels[order]


def _generate_spherical(
    counts: IntArray,
    d: int,
    separation: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray, FloatArray, dict[str, Any]]:
    k = counts.size
    centers = _circle_centers(k, d, separation)
    pieces = [rng.normal(loc=centers[j], scale=1.0, size=(int(counts[j]), d)) for j in range(k)]
    x = np.vstack(pieces)
    labels = np.repeat(np.arange(k, dtype=np.int64), counts)
    x, labels = _shuffle(x, labels, rng)
    covariances = np.repeat(np.eye(d)[None, :, :], k, axis=0)
    return x, labels, centers, {
        "separation": separation,
        "covariances": covariances,
        "count_construction": "exact_largest_remainder",
    }


def _anisotropic_transform(d: int) -> FloatArray:
    if d < 2:
        raise ValueError("anisotropic scenarios require d at least two")
    angle = np.pi / 6.0
    planar_rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    planar = planar_rotation @ np.diag([2.5, 0.4]) @ planar_rotation.T
    transform = np.eye(d)
    transform[:2, :2] = planar
    return transform


def _generate_anisotropic(
    counts: IntArray,
    d: int,
    separation: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray, FloatArray, dict[str, Any]]:
    k = counts.size
    latent_centers = _circle_centers(k, d, separation)
    latent_pieces = [
        rng.normal(loc=latent_centers[j], scale=1.0, size=(int(counts[j]), d))
        for j in range(k)
    ]
    transform = _anisotropic_transform(d)
    x = np.vstack(latent_pieces) @ transform.T
    centers = latent_centers @ transform.T
    covariance = transform @ transform.T
    covariances = np.repeat(covariance[None, :, :], k, axis=0)
    labels = np.repeat(np.arange(k, dtype=np.int64), counts)
    x, labels = _shuffle(x, labels, rng)
    return x, labels, centers, {
        "separation": separation,
        "separation_scale": "latent_mahalanobis",
        "latent_centers": latent_centers,
        "affine_transform": transform,
        "covariances": covariances,
        "covariance_condition_number": float(np.linalg.cond(covariance)),
        "count_construction": "exact_largest_remainder",
    }


def _generate_irregular(
    counts: IntArray,
    d: int,
    noise: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray, FloatArray, dict[str, Any]]:
    if counts.size != 2:
        raise ValueError("the irregular two-moons scenario requires k=2")
    if d < 2:
        raise ValueError("the irregular two-moons scenario requires d at least two")
    pieces: list[FloatArray] = []
    for cluster, count in enumerate(counts):
        angle = rng.uniform(0.0, np.pi, size=int(count))
        points = np.zeros((int(count), d), dtype=float)
        if cluster == 0:
            points[:, 0] = np.cos(angle)
            points[:, 1] = np.sin(angle)
        else:
            points[:, 0] = 1.0 - np.cos(angle)
            points[:, 1] = 0.5 - np.sin(angle)
        points += rng.normal(0.0, noise, size=points.shape)
        pieces.append(points)
    x = np.vstack(pieces)
    labels = np.repeat(np.arange(2, dtype=np.int64), counts)
    x, labels = _shuffle(x, labels, rng)
    centers = np.zeros((2, d), dtype=float)
    centers[0, :2] = [0.0, 2.0 / np.pi]
    centers[1, :2] = [1.0, 0.5 - 2.0 / np.pi]
    return x, labels, centers, {
        "noise_sd": noise,
        "shape": "two_moons",
        "angle_distribution": "iid_uniform_0_pi",
        "center_definition": "population_barycenter",
        "count_construction": "exact_largest_remainder",
    }


def generate_scenario(
    scenario: str,
    *,
    n: int = 600,
    k: int = 3,
    d: int = 2,
    imbalance: str | float = "balanced",
    overlap: str | float = "moderate",
    seed: int = 0,
) -> ScenarioData:
    """Generate one labelled scenario with exact cluster counts.

    For ``spherical`` and ``anisotropic``, a numeric ``overlap`` value is the
    adjacent center separation (larger means less overlap).  For ``irregular``
    it is the two-moons noise standard deviation (larger means more overlap).
    """

    canonical = scenario.strip().lower().replace("-", "_")
    aliases = {"gaussian": "spherical", "elliptical": "anisotropic", "moons": "irregular"}
    canonical = aliases.get(canonical, canonical)
    if canonical not in {"spherical", "anisotropic", "irregular"}:
        raise ValueError(f"unknown scenario: {scenario}")
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
    ratio, imbalance_label = _resolve_imbalance(imbalance)
    counts = _exact_counts(n, k, ratio)
    rng = np.random.default_rng(seed)

    if canonical == "irregular":
        overlap_value, overlap_label = _resolve_overlap(
            overlap, IRREGULAR_NOISE, allow_zero=True
        )
        x, labels, centers, metadata = _generate_irregular(
            counts, int(d), overlap_value, rng
        )
        overlap_parameter = "noise_sd"
    else:
        overlap_value, overlap_label = _resolve_overlap(
            overlap, GAUSSIAN_SEPARATIONS, allow_zero=False
        )
        generator = _generate_spherical if canonical == "spherical" else _generate_anisotropic
        x, labels, centers, metadata = generator(counts, int(d), overlap_value, rng)
        overlap_parameter = "adjacent_center_separation"

    metadata.update(
        {
            "imbalance_level": imbalance_label,
            "target_majority_to_minority_ratio": ratio,
            "realized_majority_to_minority_ratio": float(counts.max() / counts.min()),
            "overlap_level": overlap_label,
            "overlap_parameter": overlap_parameter,
        }
    )
    return ScenarioData(
        x=x,
        labels=labels,
        true_centers=centers,
        scenario=canonical,
        imbalance_ratio=ratio,
        overlap=overlap_label,
        seed=seed,
        metadata=metadata,
    )


def factorial_scenarios(
    *,
    scenarios: Sequence[str] = ("spherical", "anisotropic", "irregular"),
    imbalances: Sequence[str | float] = ("balanced", "moderate", "severe"),
    overlaps: Sequence[str | float] = ("low", "moderate", "high"),
    n: int = 600,
    k: int = 2,
    d: int = 2,
    seed: int = 0,
) -> list[ScenarioData]:
    """Generate the full scenario-by-imbalance-by-overlap factorial design."""

    combinations = [
        (scenario, imbalance, overlap)
        for scenario in scenarios
        for imbalance in imbalances
        for overlap in overlaps
    ]
    children = np.random.SeedSequence(seed).spawn(len(combinations))
    return [
        generate_scenario(
            scenario,
            n=n,
            k=k,
            d=d,
            imbalance=imbalance,
            overlap=overlap,
            seed=int(child.generate_state(1, dtype=np.uint32)[0]),
        )
        for (scenario, imbalance, overlap), child in zip(combinations, children, strict=True)
    ]


__all__ = [
    "GAUSSIAN_SEPARATIONS",
    "IMBALANCE_RATIOS",
    "IRREGULAR_NOISE",
    "ScenarioData",
    "factorial_scenarios",
    "generate_scenario",
]
