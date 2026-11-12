"""Regression checks for the fitting procedure actually named in the paper."""
import numpy as np
import pytest

from simulations import manuscript_engine as original
from simulations.wfcm_core import ISContext, WFCMParams, build_proposal


def _adapter():
    from simulations import algorithm1_fit
    return algorithm1_fit


def test_adapter_matches_direct_algorithm1_endpoint_and_likelihood():
    adapter = _adapter()
    x = np.r_[np.linspace(-3, -1, 18), np.linspace(1, 3, 18)][:, None]
    proposal = build_proposal(x, 2, 'defensive', 17)
    context = ISContext.draw(proposal, 512, 18)
    initial = WFCMParams(.9, np.array([[-2.], [2.]]), np.array([.35, .65]), 2.)
    options = dict(max_outer=30, epsilon=1e-5, delta=1e-5, inner_maxiter=100, K_post=100)
    expected = original.fit_fixed(x, context.samples, context.log_q, 2., 2,
        initial_sigma=.9, initial_centers=initial.centers, initial_w=initial.weights,
        seed=19, **options)
    actual = adapter.fit_algorithm1(x, 2, 2., seed=19, context=context,
        initial_params=initial, n_init=1, fit_options=options)
    np.testing.assert_array_equal(actual.free_params, expected['theta'])
    assert actual.average_nll == expected['NLL'] / len(x)
    assert actual.parameterization.weight_floor == 0
    np.testing.assert_allclose(actual.parameterization.pack(actual.params), actual.free_params,
                               atol=1e-14, rtol=1e-14)
    assert actual.raw_fit['history'] == expected['history']


@pytest.mark.parametrize('converged,failures,post_success,status,expected', [
    (True, 0, True, 0, True), (True, 0, False, 1, True),
    (False, 0, True, 0, False), (True, 1, True, 0, False),
    (True, 0, False, 2, False),
])
def test_validity_matches_november_application_policy(converged, failures, post_success, status, expected):
    assert _adapter().fit_is_clean(dict(converged=converged, optimizer_failures=failures,
        post_success=post_success, post_status=status)) is expected
