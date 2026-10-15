import numpy as np
import pytest

from simulations.data_generators import factorial_scenarios, generate_scenario


def test_spherical_generator_is_reproducible_and_has_exact_counts():
    first = generate_scenario(
        "spherical", n=101, k=3, d=2, imbalance="severe", overlap="low", seed=17
    )
    second = generate_scenario(
        "spherical", n=101, k=3, d=2, imbalance="severe", overlap="low", seed=17
    )
    different = generate_scenario(
        "spherical", n=101, k=3, d=2, imbalance="severe", overlap="low", seed=18
    )
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert not np.array_equal(first.x, different.x)
    assert first.counts.sum() == 101
    assert np.all(first.counts > 0)
    assert first.counts.max() / first.counts.min() == pytest.approx(
        first.metadata["realized_majority_to_minority_ratio"]
    )


def test_spherical_centers_have_requested_adjacent_separation():
    scenario = generate_scenario(
        "spherical", n=90, k=3, d=4, imbalance=1.0, overlap=3.25, seed=2
    )
    differences = scenario.true_centers[:, None, :] - scenario.true_centers[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    positive = distances[distances > 1e-12]
    np.testing.assert_allclose(positive, 3.25, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(scenario.counts, [30, 30, 30])


def test_anisotropic_affine_design_preserves_latent_mahalanobis_separation():
    scenario = generate_scenario(
        "anisotropic", n=120, k=3, d=2, imbalance="moderate", overlap="high", seed=4
    )
    covariance = np.asarray(scenario.metadata["covariances"])[0]
    inverse = np.linalg.inv(covariance)
    pairwise = []
    for left in range(3):
        for right in range(left):
            delta = scenario.true_centers[left] - scenario.true_centers[right]
            pairwise.append(np.sqrt(delta @ inverse @ delta))
    np.testing.assert_allclose(pairwise, scenario.metadata["separation"], rtol=1e-12)
    assert scenario.metadata["covariance_condition_number"] > 30.0


def test_noiseless_irregular_points_follow_two_moon_equations():
    scenario = generate_scenario(
        "irregular", n=100, k=2, d=2, imbalance="moderate", overlap=0.0, seed=21
    )
    first = scenario.x[scenario.labels == 0]
    second = scenario.x[scenario.labels == 1]
    np.testing.assert_allclose(first[:, 0] ** 2 + first[:, 1] ** 2, 1.0, atol=2e-15)
    np.testing.assert_allclose(
        (second[:, 0] - 1.0) ** 2 + (second[:, 1] - 0.5) ** 2,
        1.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        scenario.true_centers[:, :2], [[0.0, 2.0 / np.pi], [1.0, 0.5 - 2.0 / np.pi]]
    )


def test_factorial_grid_has_stable_unique_seeds_and_expected_size():
    kwargs = dict(
        scenarios=("spherical", "irregular"),
        imbalances=("balanced", "severe"),
        overlaps=("low", "high"),
        n=30,
        k=2,
        d=2,
        seed=22,
    )
    first = factorial_scenarios(**kwargs)
    second = factorial_scenarios(**kwargs)
    assert len(first) == 8
    assert len({item.seed for item in first}) == 8
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.x, right.x)
        assert left.to_record()["metadata"] == right.to_record()["metadata"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "unknown"},
        {"scenario": "spherical", "imbalance": 0.5},
        {"scenario": "spherical", "overlap": "unknown"},
        {"scenario": "spherical", "n": 2, "k": 3},
        {"scenario": "anisotropic", "d": 1},
        {"scenario": "irregular", "k": 3},
    ],
)
def test_invalid_scenario_controls_raise(kwargs):
    arguments = {"n": 30, "k": 2, "d": 2, "seed": 0}
    arguments.update(kwargs)
    scenario = arguments.pop("scenario")
    with pytest.raises(ValueError):
        generate_scenario(scenario, **arguments)
