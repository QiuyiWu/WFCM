import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import multivariate_t

from simulations.wfcm_core import (
    ISContext,
    DefensiveMixtureProposal,
    WFCMParameterization,
    WFCMParams,
    align_to_reference,
    average_nll_and_gradient,
    build_proposal,
    energy,
    energy_and_gradient_free,
    empirical_score_information,
    fit_wfcm,
    membership,
    model_fisher_information,
    sample_wfcm_1d,
    weighted_xie_beni,
)


def test_membership_uses_manuscript_weights():
    x = np.array([[0.0]])
    centers = np.array([[-1.0], [1.0]])
    weights = np.array([0.2, 0.8])
    u = membership(x, centers, weights, m=2.0)
    expected = np.array([[0.8, 0.2]])
    np.testing.assert_allclose(u, expected, atol=1e-12)


def test_energy_equals_weighted_fcm_profile_loss():
    x = np.array([[-0.5], [0.25], [2.0]])
    centers = np.array([[-1.0], [1.0]])
    weights = np.array([0.3, 0.7])
    m = 1.8
    sigma = 1.4
    u = membership(x, centers, weights, m)
    d2 = (x[:, None, :] - centers[None, :, :]) ** 2
    profile = np.sum(weights[None, :] * u**m * d2[:, :, 0], axis=1) / sigma**2
    np.testing.assert_allclose(energy(x, sigma, centers, weights, m), profile, rtol=2e-12)


def test_reduced_parameterization_round_trip():
    spec = WFCMParameterization(k=3, d=2, m_fixed=2.1, weight_floor=1e-4)
    params = WFCMParams(
        1.7,
        np.array([[-2.0, 0.0], [0.5, 1.0], [2.0, -1.0]]),
        np.array([0.15, 0.35, 0.5]),
        2.1,
    )
    recovered = spec.unpack(spec.pack(params))
    np.testing.assert_allclose(recovered.centers, params.centers)
    np.testing.assert_allclose(recovered.weights, params.weights)
    np.testing.assert_allclose(recovered.sigma, params.sigma)


def test_parameter_and_importance_arrays_are_immutable_copies():
    centers = np.array([[-1.0], [1.0]])
    weights = np.array([0.4, 0.6])
    params = WFCMParams(1.0, centers, weights, 2.0)
    centers[0, 0] = 99.0
    weights[:] = [0.9, 0.1]
    np.testing.assert_allclose(params.centers[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(params.weights, [0.4, 0.6])
    with pytest.raises(ValueError):
        params.weights[0] = 0.5

    samples = np.zeros((3, 1))
    log_q = np.zeros(3)
    context = ISContext(samples, log_q, "test", 1)
    samples[:] = 2.0
    log_q[:] = 4.0
    np.testing.assert_array_equal(context.samples, np.zeros((3, 1)))
    np.testing.assert_array_equal(context.log_q, np.zeros(3))


def test_energy_gradient_matches_finite_difference():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 2))
    spec = WFCMParameterization(k=2, d=2, m_fixed=None, m_bounds=(1.2, 3.0))
    params = WFCMParams(
        1.2,
        np.array([[-1.0, 0.2], [1.3, -0.1]]),
        np.array([0.35, 0.65]),
        1.9,
    )
    z = spec.pack(params)
    _, analytic = energy_and_gradient_free(z, x, spec)
    numeric = approx_derivative(
        lambda value: energy_and_gradient_free(value, x, spec)[0],
        z,
        method="3-point",
    )
    np.testing.assert_allclose(analytic, numeric, rtol=3e-5, atol=3e-6)


def test_nll_gradient_matches_finite_difference():
    rng = np.random.default_rng(6)
    x = np.r_[rng.normal(-1.0, 0.7, 30), rng.normal(1.0, 0.7, 30)][:, None]
    proposal = build_proposal(x, 2, "defensive", seed=7)
    context = ISContext.draw(proposal, 1_000, seed=8)
    spec = WFCMParameterization(k=2, d=1, m_fixed=2.0)
    z = spec.pack(WFCMParams(1.0, np.array([[-0.8], [0.9]]), np.array([0.4, 0.6]), 2.0))
    _, analytic = average_nll_and_gradient(z, x, context, spec)
    numeric = approx_derivative(
        lambda value: np.array([average_nll_and_gradient(value, x, context, spec)[0]]),
        z,
        method="3-point",
    ).reshape(-1)
    np.testing.assert_allclose(analytic, numeric, rtol=8e-5, atol=8e-6)


def test_alignment_uses_true_label_order():
    reference = np.array([[-2.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
    estimated = reference[[2, 0, 1]] + 0.01
    weights = np.array([0.5, 0.2, 0.3])
    centers_a, weights_a, indices = align_to_reference(estimated, weights, reference)
    np.testing.assert_allclose(centers_a, reference + 0.01)
    np.testing.assert_allclose(weights_a, np.array([0.2, 0.3, 0.5]))
    np.testing.assert_array_equal(indices, np.array([1, 2, 0]))


def test_independent_grid_sampler_matches_symmetric_mean():
    params = WFCMParams(
        sigma=1.0,
        centers=np.array([[-1.5], [1.5]]),
        weights=np.array([0.5, 0.5]),
        m=2.0,
    )
    draws = sample_wfcm_1d(params, size=10_000, seed=12, grid_size=50_001)
    assert abs(float(np.mean(draws))) < 0.05
    assert np.unique(draws).size > 9_900


def test_weighted_xie_beni_is_finite_for_separated_centers():
    x = np.array([[-1.1], [-0.9], [0.9], [1.1]])
    params = WFCMParams(1.0, np.array([[-1.0], [1.0]]), np.array([0.5, 0.5]), 2.0)
    value = weighted_xie_beni(x, params)
    assert np.isfinite(value)
    assert value > 0.0


def test_weighted_xie_beni_reduces_to_classical_index_for_equal_weights():
    x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    centers = np.array([[-1.25], [1.25]])
    params = WFCMParams(1.0, centers, np.array([0.5, 0.5]), 2.0)
    u = membership(x, centers, params.weights, params.m)
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    separation = float((centers[0, 0] - centers[1, 0]) ** 2)
    classical = float(np.sum(u**params.m * d2) / (x.shape[0] * separation))
    assert weighted_xie_beni(x, params) == pytest.approx(classical)


def test_defensive_student_t_logpdf_matches_scipy_for_correlated_scale():
    rng = np.random.default_rng(31)
    training = rng.normal(size=(40, 2))
    base = build_proposal(training, 2, kind="gmm_full", seed=2)
    location = np.array([0.2, -0.4])
    scale = np.array([[2.0, 0.7], [0.7, 1.3]])
    proposal = DefensiveMixtureProposal(base, location, scale, df=6.0)
    points = rng.normal(size=(12, 2))
    expected = multivariate_t.logpdf(points, loc=location, shape=scale, df=6.0)
    np.testing.assert_allclose(proposal._t_logpdf(points), expected, rtol=2e-13, atol=2e-13)


def test_parameterization_rejects_incompatible_m_values():
    fixed = WFCMParameterization(k=2, d=1, m_fixed=2.0)
    wrong_fixed = WFCMParams(1.0, np.array([[-1.0], [1.0]]), np.array([0.5, 0.5]), 1.8)
    with pytest.raises(ValueError, match="fixed m"):
        fixed.pack(wrong_fixed)

    free = WFCMParameterization(k=2, d=1, m_fixed=None, m_bounds=(1.2, 3.0))
    outside = WFCMParams(1.0, np.array([[-1.0], [1.0]]), np.array([0.5, 0.5]), 3.1)
    with pytest.raises(ValueError, match="m_bounds"):
        free.pack(outside)


def test_importance_context_validates_shape_and_log_density():
    with pytest.raises(ValueError):
        ISContext(np.zeros((3, 2)), np.zeros(2), "proposal", 1)
    with pytest.raises(ValueError):
        ISContext(np.zeros((3, 2)), np.array([0.0, np.nan, 0.0]), "proposal", 1)


def test_fit_retains_inference_context_and_fisher_inverse_is_consistent():
    rng = np.random.default_rng(44)
    x = np.r_[rng.normal(-1.5, 0.45, 30), rng.normal(1.5, 0.45, 30)][:, None]
    proposal = build_proposal(x, 2, "defensive", seed=45)
    context = ISContext.draw(proposal, 1_500, seed=46)
    fit = fit_wfcm(x, 2, m=2.0, n_init=1, maxiter=120, seed=47, context=context)
    assert fit.importance_context is context
    assert fit.parameterization is not None
    information, inverse = model_fisher_information(fit)
    np.testing.assert_allclose(information, information.T, atol=1e-12)
    np.testing.assert_allclose(inverse, inverse.T, atol=1e-12)
    np.testing.assert_allclose(information @ inverse, np.eye(information.shape[0]), atol=2e-7)
    assert np.min(np.linalg.eigvalsh(information)) > 0.0
    empirical = empirical_score_information(fit, x)
    np.testing.assert_allclose(empirical, empirical.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(empirical)) >= -1e-10
    record = fit.to_record()
    assert "importance_context" not in record and "parameterization" not in record
