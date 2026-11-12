"""November Algorithm 1, with the existing 1D simulation integration design.

Only `manuscript_engine.fit_fixed` fits parameters. The wfcm_core imports
provide density coordinates and the unchanged iid integration proposal;
neither its optimizer nor any preliminary-fit/safeguard path is called.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from simulations.manuscript_engine import fit_fixed, nll_gradient
from simulations.wfcm_core import (
    ISContext, WFCMParameterization, WFCMParams, build_proposal,
)


FIT_OPTIONS = dict(max_outer=2000, epsilon=1e-5, delta=1e-5,
                   inner_maxiter=1000, K_post=1000)


def fit_is_clean(fit):
    """The November application policy, including post iteration-limit status."""
    return bool(fit['converged'] and not fit['optimizer_failures'] and
                (fit['post_success'] or fit['post_status'] == 1))


@dataclass
class Algorithm1Fit:
    params: WFCMParams
    free_params: np.ndarray
    parameterization: WFCMParameterization
    average_nll: float
    success: bool
    raw_fit: dict
    starts: list[dict]
    importance_context: ISContext
    proposal_seed: int | None
    runtime_seconds: float
    gradient_inf_norm: float


def fit_algorithm1(x, k, m=2., *, importance_size=5000, seed=0, n_init=3,
                   initial_params=None, context=None, fit_options=None):
    """Random-observation starts, or one supplied bootstrap warm start.

    All starts share one iid importance cloud. As in application `select_m`,
    choose the lowest finite NLL, then report its actual convergence status;
    do not rescue a failed selected endpoint with another optimization path.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= k <= len(x):
        raise ValueError('Expected finite data and 1 <= k <= n')
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError('n_init must be a positive integer')
    if initial_params is not None and n_init != 1:
        raise ValueError('A bootstrap warm start must be the sole initial value')
    options = dict(FIT_OPTIONS)
    if fit_options is not None:
        options.update(fit_options)
    begin = perf_counter()
    rng = np.random.default_rng(seed)
    proposal_seed = None
    if context is None:
        proposal_seed = int(rng.integers(0, 2**31-1))
        importance_seed = int(rng.integers(0, 2**31-1))
        proposal = build_proposal(x, k, 'defensive', proposal_seed)
        context = ISContext.draw(proposal, importance_size, importance_seed)
    starts = [int(seed)] if n_init == 1 else [int(s) for s in rng.integers(0, 2**31-1, n_init)]
    initial = {}
    if initial_params is not None:
        initial_params.validate()
        if initial_params.k != k or initial_params.d != x.shape[1] or initial_params.m != m:
            raise ValueError('Initial parameters do not match fixed k, d, m')
        initial = dict(initial_sigma=initial_params.sigma,
                       initial_centers=initial_params.centers,
                       initial_w=initial_params.weights)
    fits = [fit_fixed(x, context.samples, context.log_q, m, k, seed=s,
                      **initial, **options) for s in starts]
    best = min(fits, key=lambda f: f['NLL'])
    params = WFCMParams(best['sigma'], best['centers'], best['w'], m)
    spec = WFCMParameterization(k, x.shape[1], m_fixed=m, weight_floor=0.)
    _, grad = nll_gradient(best['theta'], x, context.samples, context.log_q, m, k)
    return Algorithm1Fit(params, best['theta'].copy(), spec, best['NLL']/len(x),
                         fit_is_clean(best), best, fits, context, proposal_seed,
                         perf_counter()-begin, float(np.max(np.abs(grad)))/len(x))


def fit_record(fit, *, history=False):
    """Persist acceptance evidence and endpoints without saving full clouds."""
    raw = fit.raw_fit
    out = {key: raw[key] for key in (
        'sigma', 'centers', 'w', 'm', 'k', 'NLL', 'theta', 'converged',
        'optimizer_failures', 'post_success', 'post_status', 'post_message',
        'post_iterations', 'post_attempts', 'seed', 'n_iter', 'fit_options',
        'IS_diagnostics')}
    out.update(success=fit.success, average_nll=fit.average_nll,
               raw_gradient_per_observation=fit.gradient_inf_norm,
               importance_seed=fit.importance_context.seed,
               proposal_seed=fit.proposal_seed,
               proposal_name=fit.importance_context.proposal_name,
               starts=[{key: f[key] for key in ('seed', 'NLL', 'converged',
                   'optimizer_failures', 'post_success', 'post_status')} for f in fit.starts])
    if history or not fit.success:
        out['history'] = raw['history']
    return out
