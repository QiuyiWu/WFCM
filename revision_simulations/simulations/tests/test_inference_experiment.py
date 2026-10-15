from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

import simulations.inference_experiment as inference_experiment
from simulations.inference_experiment import (
    _log_normalizer_grid_1d,
    _lrt_one,
    _null_embedded_start,
    Profile,
    exact_loglik_1d,
    fit_alternative_for_lrt,
    normal_null_fit,
)
from simulations.wfcm_core import (
    ISContext,
    WFCMParameterization,
    WFCMParams,
    average_nll_and_gradient,
    build_proposal,
    energy,
)


def test_grid_normalizer_matches_k1_gaussian_constant():
    params = WFCMParams(
        sigma=1.7,
        centers=np.array([[0.4]]),
        weights=np.array([1.0]),
        m=2.0,
    )
    expected = np.log(np.sqrt(np.pi) * params.sigma)
    np.testing.assert_allclose(_log_normalizer_grid_1d(params), expected, atol=2e-8)


def test_coincident_k2_represents_same_k1_density_after_scale_change():
    m = 1.8
    k1 = WFCMParams(1.3, np.array([[0.2]]), np.array([1.0]), m)
    weights = np.array([0.4, 0.6])
    b = np.sum(weights ** (-1.0 / (m - 1.0)))
    k2 = WFCMParams(
        k1.sigma / b ** ((m - 1.0) / 2.0),
        np.array([[0.2], [0.2]]),
        weights,
        m,
    )
    x = np.linspace(-3.0, 3.0, 31)[:, None]
    np.testing.assert_allclose(
        energy(x, k1.sigma, k1.centers, k1.weights, m),
        energy(x, k2.sigma, k2.centers, k2.weights, m),
        atol=2e-12,
    )
    np.testing.assert_allclose(exact_loglik_1d(x, k1), exact_loglik_1d(x, k2), atol=2e-7)


def test_k1_exact_loglik_uses_the_normal_scale_convention():
    x = np.array([-1.4, -0.1, 0.2, 1.7])[:, None]
    mean = 0.3
    sd = 1.2
    params = WFCMParams(
        sigma=np.sqrt(2.0) * sd,
        centers=np.array([[mean]]),
        weights=np.array([1.0]),
        m=3.1,
    )
    expected = np.sum(norm.logpdf(x[:, 0], loc=mean, scale=sd))
    np.testing.assert_allclose(exact_loglik_1d(x, params), expected, atol=2e-12)


def test_analytic_null_fit_is_the_exact_gaussian_mle():
    x = np.array([-1.4, -0.1, 0.2, 1.7])[:, None]
    mean, sd, loglik = normal_null_fit(x)
    assert mean == pytest.approx(float(np.mean(x)))
    assert sd == pytest.approx(float(np.sqrt(np.mean((x[:, 0] - mean) ** 2))))
    assert loglik == pytest.approx(
        float(np.sum(norm.logpdf(x[:, 0], loc=mean, scale=sd)))
    )


def test_analytic_null_fit_rejects_degenerate_sample_instead_of_flooring_scale():
    with pytest.raises(ValueError, match="zero or nonfinite scale"):
        normal_null_fit(np.ones((5, 1)))


def test_grid_normalizer_refines_narrow_widely_separated_modes():
    params = WFCMParams(
        sigma=1.0,
        centers=np.array([[0.0], [80_000.0]]),
        weights=np.array([0.5, 0.5]),
        m=2.0,
    )
    # At this separation the two modes are independent to better than the
    # requested tolerance.  A global 1001-point grid misses them; the local
    # refinement must recover both Gaussian-mode integrals.
    separated_limit = np.log(2.0 * np.sqrt(np.pi / 0.5))
    np.testing.assert_allclose(
        _log_normalizer_grid_1d(params, grid_size=1_001),
        separated_limit,
        atol=2e-8,
    )


def test_grid_normalizer_rejects_invalid_resolution():
    params = WFCMParams(1.0, np.array([[0.0]]), np.array([1.0]), 2.0)
    with pytest.raises(ValueError):
        _log_normalizer_grid_1d(params, grid_size=100)
    with pytest.raises(ValueError):
        _log_normalizer_grid_1d(params, grid_size=True)


def test_null_embedding_uses_exact_scale_for_arbitrary_fixed_m():
    null = WFCMParams(1.4, np.array([[-0.3]]), np.array([1.0]), 2.7)
    x = np.linspace(-4.0, 4.0, 101)[:, None]
    embedded = _null_embedded_start(x, null.m, null, split_fraction=0.0)
    np.testing.assert_allclose(
        energy(x, null.sigma, null.centers, null.weights, null.m),
        energy(
            x,
            embedded.sigma,
            embedded.centers,
            embedded.weights,
            embedded.m,
        ),
        rtol=2e-13,
        atol=2e-13,
    )


def test_null_embedding_nests_the_common_cloud_finite_is_objective():
    x = np.random.default_rng(81).normal(size=(60, 1))
    null = WFCMParams(1.2, np.array([[0.1]]), np.array([1.0]), 2.3)
    embedded = _null_embedded_start(x, null.m, null, split_fraction=0.0)
    proposal = build_proposal(x, 2, "defensive", seed=82)
    context = ISContext.draw(proposal, size=2_001, seed=83)
    spec1 = WFCMParameterization(1, 1, m_fixed=null.m)
    spec2 = WFCMParameterization(2, 1, m_fixed=null.m)
    value1, _ = average_nll_and_gradient(spec1.pack(null), x, context, spec1)
    value2, _ = average_nll_and_gradient(
        spec2.pack(embedded),
        x,
        context,
        spec2,
    )
    np.testing.assert_allclose(value1, value2, rtol=2e-13, atol=2e-13)


def test_lrt_retains_and_reports_negative_untruncated_candidate(monkeypatch):
    monkeypatch.setattr(
        inference_experiment,
        "normal_null_fit",
        lambda _x: (0.0, 1.0, 10.0),
    )
    monkeypatch.setattr(
        inference_experiment,
        "exact_loglik_1d",
        lambda _x, _params: 9.75,
    )
    x = np.random.default_rng(91).normal(size=(50, 1))
    _, _, statistic, raw = fit_alternative_for_lrt(
        x,
        m=2.0,
        seed=92,
        m_is=1_001,
        n_init=1,
        maxiter=100,
    )
    assert raw == -0.5
    assert statistic == 0.0


def test_lrt_rejects_nonfinite_deterministic_score(monkeypatch):
    monkeypatch.setattr(
        inference_experiment,
        "normal_null_fit",
        lambda _x: (0.0, 1.0, 10.0),
    )
    monkeypatch.setattr(
        inference_experiment,
        "exact_loglik_1d",
        lambda _x, _params: np.nan,
    )
    x = np.random.default_rng(94).normal(size=(40, 1))
    with pytest.raises(FloatingPointError, match="nonfinite deterministic LRT"):
        fit_alternative_for_lrt(
            x,
            m=2.0,
            seed=95,
            m_is=1_001,
            n_init=1,
            maxiter=100,
        )


def test_lrt_repeats_n_init_and_excludes_failed_raw_diagnostics(monkeypatch):
    profile = Profile(
        coverage_reps=1,
        bootstrap_reps=1,
        lrt_reps=1,
        lrt_bootstrap_reps=6,
        selection_reps=1,
        coverage_ns=(50,),
        lrt_ns=(50,),
        selection_ns=(50,),
        fit_n_init=7,
        maxiter=20,
        fisher_draws=100,
    )
    calls = []
    # The first value is observed.  The second is a finite but failed
    # bootstrap refit and must not contaminate the negative-raw diagnostic.
    raw_values = iter([-0.01, -100.0, -0.2, 0.0, 0.1, 0.2, 0.3])
    fitting_raw_values = iter([0.01, -100.0, 0.2, 0.0, 0.1, 0.2, 0.3])

    def fake_fit(_x, *, n_init, **_kwargs):
        call = len(calls)
        calls.append(n_init)
        raw = next(raw_values)
        fitting_raw = next(fitting_raw_values)
        success = call != 1
        alt = SimpleNamespace(
            success=success,
            gradient_inf_norm=0.0,
            average_nll=1.0 - fitting_raw / (2.0 * 50.0),
            diagnostics={"ess_fraction": 1.0},
        )
        null = SimpleNamespace(
            success=success,
            gradient_inf_norm=0.0,
            average_nll=1.0,
            exact_loglik=-70.0,
            params=WFCMParams(
                np.sqrt(2.0),
                np.array([[0.0]]),
                np.array([1.0]),
                2.0,
            ),
        )
        return alt, null, max(0.0, raw), raw

    monkeypatch.setattr(
        inference_experiment,
        "fit_alternative_for_lrt",
        fake_fit,
    )
    row = _lrt_one(profile, n=50, separation=0.0, replicate=0, base_seed=93)
    assert row["statistic_name"] == (
        "bootstrap-calibrated implemented approximate likelihood-ratio statistic"
    )
    assert calls == [profile.fit_n_init] * (1 + profile.lrt_bootstrap_reps)
    assert row["embedded_null_fallback_used"]
    assert row["bootstrap_finite"] == 6
    assert row["bootstrap_valid"] == 5
    assert row["bootstrap_negative_raw_fraction"] == pytest.approx(1.0 / 5.0)
    assert row["bootstrap_minimum_raw_statistic"] == -0.2
    assert row["fitting_objective_raw_statistic"] == pytest.approx(0.01)
    assert row["fitting_objective_nested_within_tolerance"]
    assert row["cross_objective_fallback_used"]
    assert row["material_cross_objective_mismatch"]
    assert row["bootstrap_cross_objective_fallback_fraction"] == pytest.approx(
        1.0 / 5.0
    )
    assert row["bootstrap_cross_objective_fallback_count"] == 1
    assert row["bootstrap_material_cross_objective_mismatch_fraction"] == pytest.approx(
        1.0 / 5.0
    )
    assert row["bootstrap_material_cross_objective_mismatch_count"] == 1
    assert row["bootstrap_cross_objective_diagnostic_defined"] == 5
    assert row["bootstrap_fitting_objective_negative_fraction"] == 0.0
    assert row["bootstrap_minimum_fitting_objective_raw_statistic"] == pytest.approx(
        0.0
    )
    assert row["alternative_n_init"] == profile.fit_n_init
    assert row["null_n_init"] == 0
    assert row["null_estimator"] == "analytic_gaussian_mle"
    assert row["null_scoring_method"] == "closed_form_gaussian_loglik"
    assert not row["null_optimizer_used"]
    assert row["alternative_candidate_rule"] == (
        "max_fitted_alternative_and_embedded_analytic_null"
    )
    assert row["scored_candidate_selected"] == "embedded_analytic_null"
    assert row["analytic_null_loglik"] == -70.0
    assert row["fitted_alternative_loglik"] == pytest.approx(-70.005)


def test_lrt_analytic_null_failure_is_a_structured_outer_failure(monkeypatch):
    profile = Profile(
        coverage_reps=1,
        bootstrap_reps=1,
        lrt_reps=1,
        lrt_bootstrap_reps=5,
        selection_reps=1,
        coverage_ns=(30,),
        lrt_ns=(30,),
        selection_ns=(30,),
        fit_n_init=2,
        maxiter=20,
        fisher_draws=100,
    )

    def fail_analytic_null(*_args, **_kwargs):
        raise ValueError("analytic Gaussian null MLE has zero scale")

    monkeypatch.setattr(
        inference_experiment,
        "fit_alternative_for_lrt",
        fail_analytic_null,
    )
    row = _lrt_one(profile, n=30, separation=0.0, replicate=0, base_seed=96)
    assert not row["analysis_success"]
    assert not row["fit_success"]
    assert not row["null_fit_success"]
    assert row["exception_type"] == "ValueError"
    assert "analytic Gaussian null MLE has zero scale" in row["failure_message"]
    assert np.isnan(row["statistic"])
    assert row["scored_candidate_selected"] is None
    assert row["bootstrap_valid"] == 0
    assert row["bootstrap_cross_objective_diagnostic_defined"] == 0
    assert np.isnan(row["bootstrap_material_cross_objective_mismatch_fraction"])
