import json

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from simulations.competitors import (
    DEFAULT_METHODS,
    fcm_membership,
    fit_competitor,
    fit_competitors,
    polynomial_fuzzifier,
    polynomial_membership,
)


def _separated_data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.r_[rng.normal((-2.5, 0.0), 0.25, (25, 2)), rng.normal((2.5, 0.0), 0.25, (25, 2))]
    labels = np.repeat([0, 1], 25)
    return x, labels


def test_fcm_membership_formula_and_zero_distance_convention():
    x = np.array([[0.0], [1.0]])
    centers = np.array([[-1.0], [2.0]])
    memberships = fcm_membership(x, centers, m=2.0)
    np.testing.assert_allclose(memberships[0], [0.8, 0.2])

    duplicate_centers = np.array([[1.0], [1.0], [3.0]])
    at_center = fcm_membership(np.array([[1.0]]), duplicate_centers, m=2.0)
    np.testing.assert_allclose(at_center, [[0.5, 0.5, 0.0]])


def test_polynomial_membership_known_values_and_beta_zero_limit():
    x = np.array([[0.0]])
    centers = np.array([[1.0], [np.sqrt(2.0)]])
    np.testing.assert_allclose(polynomial_membership(x, centers, beta=0.2), [[0.75, 0.25]])
    np.testing.assert_allclose(polynomial_membership(x, centers, beta=0.5), [[1.0, 0.0]])

    rng = np.random.default_rng(12)
    data = rng.normal(size=(20, 2))
    prototypes = rng.normal(size=(3, 2))
    np.testing.assert_allclose(
        polynomial_membership(data, prototypes, beta=0.0),
        fcm_membership(data, prototypes, m=2.0),
        rtol=2e-13,
        atol=2e-14,
    )


def test_polynomial_fuzzy_fit_is_monotone_sparse_and_reproducible():
    x, truth = _separated_data(3)
    first = fit_competitor(
        "polynomial_fcm", x, 2, seed=14, beta=0.5, n_init=3, max_iter=100
    )
    second = fit_competitor(
        "polynomial_fcm", x, 2, seed=14, beta=0.5, n_init=3, max_iter=100
    )
    assert first.success and second.success
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_allclose(first.centers, second.centers)
    np.testing.assert_allclose(first.memberships, second.memberships)
    assert adjusted_rand_score(truth, first.labels) == pytest.approx(1.0)
    history = np.asarray(first.diagnostics["objective_history"])
    assert np.all(np.diff(history) <= 1e-9 * (1.0 + np.abs(history[:-1])))
    assert first.diagnostics["zero_membership_fraction"] > 0.0


def test_fcm_fit_recovers_separated_clusters_and_has_valid_memberships():
    x, truth = _separated_data(5)
    result = fit_competitor("fcm", x, 2, seed=8, n_init=3, max_iter=100)
    assert result.success
    assert adjusted_rand_score(truth, result.labels) == pytest.approx(1.0)
    np.testing.assert_allclose(result.memberships.sum(axis=1), 1.0)
    assert np.all(np.diff(result.diagnostics["objective_history"]) <= 1e-8)


def test_ekmeans_wrapper_uses_official_estimator_output():
    sklekmeans = pytest.importorskip("sklekmeans")
    x, _ = _separated_data(6)
    result = fit_competitor("ekmeans", x, 2, seed=11, n_init=2, max_iter=80)
    direct = sklekmeans.EKMeans(
        n_clusters=2,
        alpha="dvariance",
        scale=2.0,
        n_init=2,
        max_iter=80,
        random_state=11,
        use_numba=False,
    ).fit(x)
    assert result.success
    np.testing.assert_array_equal(result.labels, direct.labels_)
    np.testing.assert_allclose(result.centers, direct.cluster_centers_)
    np.testing.assert_allclose(result.memberships, direct.U_)
    assert result.diagnostics["iterations"] == direct.n_iter_ + 1
    assert result.diagnostics["package"] == "sklekmeans"


@pytest.mark.parametrize("method", DEFAULT_METHODS)
def test_all_competitors_return_unified_valid_schema(method):
    x, _ = _separated_data(9)
    options = {"n_init": 2, "max_iter": 80} if method in {
        "fcm",
        "kmeans",
        "ekmeans",
        "polynomial_fcm",
        "gmm_full",
    } else {}
    result = fit_competitor(method, x, 2, seed=13, **options)
    assert result.runtime_seconds >= 0.0
    assert result.labels.shape == (x.shape[0],)
    assert result.centers.ndim == 2 and result.centers.shape[1] == x.shape[1]
    assert result.n_clusters_found <= result.centers.shape[0]
    assert result.diagnostics["n_centers_returned"] == result.centers.shape[0]
    assert result.success, result.message
    assert 0.0 <= result.noise_fraction <= 1.0
    record = result.to_record(include_assignments=True)
    assert len(record["labels"]) == x.shape[0]


def test_hdbscan_does_not_use_requested_k_as_a_tuning_parameter():
    x, _ = _separated_data(2)
    result_two = fit_competitor("hdbscan", x, 2, seed=1, min_cluster_size=5)
    result_four = fit_competitor("hdbscan", x, 4, seed=999, min_cluster_size=5)
    np.testing.assert_array_equal(result_two.labels, result_four.labels)
    np.testing.assert_allclose(result_two.centers, result_four.centers)
    assert result_two.diagnostics["k_used_by_algorithm"] is False


def test_hdbscan_all_noise_is_valid_and_serializes_without_nan():
    result = fit_competitor(
        "hdbscan", np.eye(8), 2, seed=1, min_cluster_size=8, min_samples=8
    )
    assert result.success
    assert result.n_clusters_found == 0
    assert result.noise_fraction == 1.0
    np.testing.assert_array_equal(result.labels, -np.ones(8, dtype=int))
    json.dumps(result.to_record(include_assignments=True), allow_nan=False)


def test_dispatcher_records_failure_and_alias_options_are_honored():
    bad = fit_competitor("fcm", np.array([[0.0], [np.nan]]), 2, seed=1)
    assert not bad.success
    np.testing.assert_array_equal(bad.labels, [-1, -1])
    assert bad.n_clusters_found == 0
    assert "ValueError" in bad.message

    x, _ = _separated_data(1)
    results = fit_competitors(
        x,
        2,
        methods=("poly_fcm",),
        seed=4,
        method_options={"polynomial_fcm": {"beta": 0.25, "n_init": 1}},
    )
    assert results["polynomial_fcm"].diagnostics["beta"] == pytest.approx(0.25)


def test_invalid_fuzzifier_controls_are_rejected():
    with pytest.raises(ValueError):
        polynomial_fuzzifier([0.5, 0.5], beta=1.0)
    failed = fit_competitor("fcm", np.arange(6.0)[:, None], 2, m=1.0)
    assert not failed.success
