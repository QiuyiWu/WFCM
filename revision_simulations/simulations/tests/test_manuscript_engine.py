"""Tests target density, gradients and Algorithm 1 behavior, not source text."""
import importlib.util
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative


def engine():
    assert importlib.util.find_spec('simulations.manuscript_engine') is not None, 'Extracted manuscript engine is missing'
    from simulations import manuscript_engine
    return manuscript_engine


def test_sampler_known_one_center_gaussian_and_independent_seed():
    e = engine()
    truth = dict(sigma=2., centers=np.array([[1., -2.]]), w=np.array([1.]), m=2.)
    x = e.sample_iid(truth, 30000, seed=112)
    assert x.shape == (30000, 2)
    np.testing.assert_allclose(x.mean(axis=0), [1., -2.], atol=.035)
    np.testing.assert_allclose(np.cov(x.T), 2*np.eye(2), atol=.06)
    assert abs(np.corrcoef(x[:-1, 0], x[1:, 0])[0, 1]) < .025
    np.testing.assert_array_equal(x[:5], e.sample_iid(truth, 30000, seed=112)[:5])
    assert not np.array_equal(x[:5], e.sample_iid(truth, 30000, seed=113)[:5])


def test_nll_gradient_matches_finite_difference_and_existing_core():
    e = engine()
    from simulations.wfcm_core import WFCMParams, WFCMParameterization, average_nll_and_gradient, ISContext
    rng = np.random.default_rng(91)
    x, z = rng.normal(size=(80, 2)), rng.normal(size=(2000, 2))*3
    logq = -np.log(18*np.pi)-np.sum(z*z, axis=1)/18
    centers, weights = np.array([[-1., .2], [1., -.1]]), np.array([.3, .7])
    theta = e._pack(.9, centers, weights)
    val, grad = e.nll_gradient(theta, x, z, logq, 2., 2)
    fd = approx_derivative(lambda t: e.nll_gradient(t, x, z, logq, 2., 2)[0], theta).ravel()
    np.testing.assert_allclose(grad, fd, rtol=2e-5, atol=2e-5)
    spec = WFCMParameterization(2, 2, m_fixed=2., weight_floor=0.)
    context = ISContext(z, logq, 'fixture', 1)
    other, other_grad = average_nll_and_gradient(spec.pack(WFCMParams(.9, centers, weights, 2.)), x, context, spec)
    np.testing.assert_allclose(val/len(x), other, rtol=1e-11)
    np.testing.assert_allclose(grad/len(x), other_grad, atol=1e-10)


def test_blockwise_centroid_step_and_refinement_do_not_silently_disappear():
    e = engine()
    truth = dict(sigma=.6, centers=np.array([[-2.], [2.]]), w=np.array([.5, .5]), m=2.)
    x = e.sample_iid(truth, 100, 21)
    z, logq, _ = e.build_proposal(x, num_samples=2048, n_components=2, seed=2)
    initial = np.array([[-1.8], [1.8]])
    u = e.memberships(x, initial, np.array([.5,.5]), 2.)
    expected = ((u*u).T@x)/(u*u).sum(axis=0)[:,None]
    first = e.fit_fixed(x,z,logq,2.,2,initial_centers=initial,initial_sigma=.6,max_outer=1,K_post=0)
    np.testing.assert_allclose(first['centers'], expected, atol=1e-12)
    fitted = e.fit_fixed(x,z,logq,2.,2,initial_centers=initial,initial_sigma=.6,max_outer=30,K_post=100)
    stages = [h['stage'] for h in fitted['history']]
    assert 'blockwise' in stages and 'joint_refinement' in stages
    last_block = [h['NLL'] for h in fitted['history'] if h['stage']=='blockwise'][-1]
    assert fitted['NLL'] <= last_block+1e-7


def test_sampler_budget_failure_never_returns_short_dataset():
    e = engine()
    with pytest.raises(RuntimeError, match='budget'):
        e.sample_iid(dict(sigma=1., centers=np.array([[-2.],[2.]]),w=np.array([.3,.7]),m=2.),100,1,max_proposals=1)
