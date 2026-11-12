"""Numerical routines extracted unchanged from the 8 November application code.

Source: FCM_breast.ipynb at 5d15846. Same fit_fixed and sample_iid as
FCM_cell.ipynb; the proposal uses the existing scrambled-Sobol variant.
Application notebooks remain untouched.
"""

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.stats import qmc, norm, chi2, multivariate_t
from sklearn.mixture import GaussianMixture

def _data(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError('Expected a finite, nonempty samples-by-features matrix.')
    return X

class DefensiveProposal:
    """Data GMM + isotropic coverage at its centers + heavy-tailed defense.

    The isotropic component prevents thin data-covariance directions from
    undersampling the model's isotropic Gaussian tails. All densities are
    evaluated as the complete mixture, including the Student-t component.
    """
    def __init__(self, X, n_components, seed):
        self.seed = int(seed)
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                                   random_state=self.seed).fit(X)
        if not self.gmm.converged_:
            raise RuntimeError('Importance-proposal GMM did not converge.')
        self.loc = X.mean(axis=0)
        self.d = X.shape[1]
        self.variance = max(float(np.var(X,axis=0,ddof=1).mean()), 1e-8)
        self.shape = 4.*self.variance*np.eye(self.d)
        self.df = 5.
        J = len(self.gmm.weights_)
        self.weights = np.r_[.2*self.gmm.weights_, np.full(J,.7/J), .1]
        self.name = '0.2 data GMM + 0.7 isotropic center mixture + 0.1 Student-t(df=5); scrambled Sobol'
    def sample(self, n):
        if int(n) != n or n < 2:
            raise ValueError('At least two integer integration samples are required.')
        n = int(n)
        U = qmc.Sobol(d=self.d+2, scramble=True, seed=self.seed).random_base2(int(np.ceil(np.log2(n))))[:n]
        U = np.clip(U, np.finfo(float).eps, 1-np.finfo(float).eps)
        labels = np.searchsorted(np.cumsum(self.weights), U[:,0], side='right')
        G = norm.ppf(U[:,1:self.d+1]);Z = np.empty((n,self.d));J=len(self.gmm.weights_)
        for j in range(J):
            use = labels == j
            Z[use] = self.gmm.means_[j] + G[use] @ np.linalg.cholesky(self.gmm.covariances_[j]).T
            use = labels == J+j
            Z[use] = self.gmm.means_[j] + G[use]*np.sqrt(self.variance)
        use = labels == 2*J
        radial = np.sqrt(self.df / chi2.ppf(U[use,-1], self.df))
        Z[use] = self.loc + G[use]*(2*np.sqrt(self.variance))*radial[:,None]
        return Z, labels
    def score_samples(self, Z):
        d2=((Z[:,None,:]-self.gmm.means_[None,:,:])**2).sum(axis=2)
        log_iso=logsumexp(-.5*d2/self.variance,axis=1)-np.log(len(self.gmm.weights_))-.5*self.d*np.log(2*np.pi*self.variance)
        log_t=multivariate_t.logpdf(Z,loc=self.loc,shape=self.shape,df=self.df)
        return logsumexp(np.vstack([np.log(.2)+self.gmm.score_samples(Z),
                                   np.log(.7)+log_iso,np.log(.1)+log_t]),axis=0)

def build_proposal(X, num_samples=32768, n_components=None, seed=42):
    X = np.asarray(X,dtype=float)
    if X.ndim != 2 or X.shape[0]<2 or X.shape[1]<1 or not np.isfinite(X).all():
        raise ValueError('Proposal requires finite data with at least two rows and one feature.')
    components = min(4,len(X)) if n_components is None else n_components
    proposal = DefensiveProposal(X,components,seed)
    Z,_ = proposal.sample(num_samples)
    return Z,proposal.score_samples(Z),proposal

def _unpack(theta, k, d):
    centers = theta[1:1+k*d].reshape(k, d)
    logits = np.r_[theta[1+k*d:], 0.0]
    return theta[0], centers, logits-logsumexp(logits)

def _pack(sigma, centers, w):
    w = np.asarray(w, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0 or np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError('The density requires positive scale and strictly positive weights.')
    w = w/w.sum()
    return np.r_[np.log(sigma), np.asarray(centers).ravel(), np.log(w[:-1])-np.log(w[-1])]

def _terms(X, centers, logw, m):
    # Use log space; only distances at floating-point zero receive a floor.
    d2 = np.maximum(cdist(X, centers, 'sqeuclidean'), 1e-200)
    scores = -(np.log(d2)+logw)/(m-1)
    lse = logsumexp(scores, axis=1)
    return np.exp(scores-lse[:, None]), np.exp(-(m-1)*lse), d2

def memberships(X, centers, w, m):
    if m <= 1:
        raise ValueError('m must exceed 1.')
    return _terms(_data(X), np.asarray(centers), np.log(w), m)[0]

def nll_gradient(theta, X, Z, logq, m, k, compute_center_gradient=True):
    """IS NLL and analytic gradient in log-scale / center / weight-logit coordinates."""
    logsigma, centers, logw = _unpack(theta, k, X.shape[1])
    w = np.exp(logw)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        ux, hx, dx = _terms(X, centers, logw, m)
        uz, hz, dz = _terms(Z, centers, logw, m)
        ex, ez = hx*np.exp(-2*logsigma), hz*np.exp(-2*logsigma)
        lr = -ez-logq
        lse = logsumexp(lr)
        prob = np.exp(lr-lse)
        value = len(X)*(lse-np.log(len(Z))) + ex.sum()

        def gradient(A, U, e, d2, a):
            t = a[:, None]*e[:, None]*U/d2
            # Accumulate displacements directly: subtracting two large matrix
            # products loses precision when a center nearly equals a data point.
            if compute_center_gradient:
                displacement = centers[None, :, :]-A[:, None, :]
                gv = 2*np.einsum('ij,ijd->jd', t, displacement)
            else:
                gv = np.zeros_like(centers)
            gw = ((a*e)[:, None]*(U-w)).sum(axis=0)[:-1]
            return np.r_[-2*np.dot(a, e), gv.ravel(), gw]

        grad = gradient(X, ux, ex, dx, np.ones(len(X)))-gradient(Z, uz, ez, dz, len(X)*prob)
    if not np.isfinite(value) or not np.isfinite(grad).all():
        return np.inf, np.zeros_like(theta)
    return float(value), grad

def importance_diagnostics(fit, Z, logq):
    _, h, _ = _terms(Z, fit['centers'], np.log(fit['w']), fit['m'])
    lr = -h/fit['sigma']**2-logq
    lse = logsumexp(lr)
    p = np.exp(lr-lse)
    return dict(logZ=float(lse-np.log(len(Z))), ESS=float(1/np.sum(p*p)),
                ESS_fraction=float(1/(len(Z)*np.sum(p*p))), largest_IS_weight=float(p.max()))

def fit_fixed(X, Z, logq, m, k, initial_sigma=1., initial_centers=None,
              initial_w=None, seed=321, max_outer=200, epsilon=1e-5,
              delta=1e-5, inner_maxiter=1000, K_post=1000):
    """Algorithm emfuzzy for one m: block updates, then joint refinement.

    max_outer is an explicit operational guard; hitting it is reported as
    nonconvergence, never silently treated as satisfying the stopping rule.
    The weight softmax enforces the simplex throughout optimization.
    """
    X, Z = _data(X), _data(Z)
    logq = np.asarray(logq, dtype=float)
    if (m <= 1 or not np.isfinite(m) or not 1 <= k <= len(X)
            or X.shape[1] != Z.shape[1] or logq.shape != (len(Z),)
            or not np.isfinite(logq).all() or max_outer < 1 or inner_maxiter < 1
            or K_post < 0 or epsilon < 0 or delta < 0):
        raise ValueError('Invalid model, proposal, or optimizer configuration.')
    d = X.shape[1]
    centers = (X[np.random.default_rng(seed).choice(len(X), k, replace=False)].copy()
               if initial_centers is None else np.array(initial_centers, dtype=float, copy=True))
    if centers.shape != (k, d):
        raise ValueError('Initial centers must have shape (k,d).')
    w = np.full(k, 1/k) if initial_w is None else np.asarray(initial_w, dtype=float)
    if w.shape != (k,):
        raise ValueError('Initial weights must have shape (k,).')
    theta = _pack(initial_sigma, centers, w)
    value, _ = nll_gradient(theta, X, Z, logq, m, k)
    if not np.isfinite(value):
        raise RuntimeError('Initial likelihood is not finite.')
    sw_indices = np.r_[0, np.arange(1+k*d, len(theta))]
    history, converged, failures = [], False, 0

    def physical(t):
        ls, v, lw = _unpack(t, k, d)
        return np.r_[np.exp(ls), v.ravel(), np.exp(lw)]

    for iteration in range(1, max_outer+1):
        old_theta, old_value = theta.copy(), value
        _, v, lw = _unpack(theta, k, d)
        U, _, _ = _terms(X, v, lw, m)
        # The centroid update is unconditional, exactly as in the manuscript.
        logum = m*np.log(np.maximum(U, np.finfo(float).tiny))
        um = np.exp(logum-logsumexp(logum, axis=0))
        theta[1:1+k*d] = (um.T@X).ravel()

        def block_objective(sw):
            candidate = theta.copy()
            candidate[sw_indices] = sw
            f, g = nll_gradient(candidate, X, Z, logq, m, k, compute_center_gradient=False)
            return f, g[sw_indices]

        opt = minimize(block_objective, theta[sw_indices], jac=True, method='L-BFGS-B',
                       options=dict(maxiter=inner_maxiter, ftol=1e-12, gtol=1e-6, maxls=100))
        if not opt.success:
            failures += 1
        if not np.isfinite(opt.fun):
            raise RuntimeError('Nonfinite scale/weight optimization.')
        theta[sw_indices] = opt.x
        value, _ = nll_gradient(theta, X, Z, logq, m, k)
        step = float(np.linalg.norm(physical(theta)-physical(old_theta)))
        change = abs(value-old_value)
        history.append(dict(stage='blockwise', iteration=iteration, NLL=value,
                            parameter_change=step, NLL_change=change,
                            optimizer_success=bool(opt.success), message=str(opt.message)))
        converged = bool(opt.success and (step <= epsilon or change <= delta))
        if converged:
            break

    post_success, post_message, post_iterations, post_status = True, 'K_post=0: refinement skipped', 0, 0
    post_attempts = 0
    # A fresh L-BFGS history can recover from an abnormal line search. Both
    # attempts share the manuscript's TOTAL K_post iteration budget.
    if K_post:
        post_success = False
        for attempt, line_search_budget in enumerate((100, 200), start=1):
            remaining = K_post-post_iterations
            if remaining <= 0:
                break
            opt = minimize(nll_gradient, theta, args=(X, Z, logq, m, k), jac=True,
                           method='L-BFGS-B', options=dict(maxiter=remaining,
                           ftol=1e-12, gtol=1e-6, maxls=line_search_budget))
            trial_value, trial_gradient = nll_gradient(opt.x, X, Z, logq, m, k)
            accepted = bool(np.isfinite(trial_value) and trial_value <= value+1e-7)
            if accepted:
                theta, value = opt.x.copy(), float(trial_value)
            post_iterations += int(opt.nit)
            post_attempts = attempt
            post_success = bool(opt.success and accepted)
            post_status = int(opt.status) if accepted else 2
            post_message = str(opt.message) if accepted else 'Refinement returned a nonfinite or worsening point.'
            history.append(dict(stage='joint_refinement', iteration=post_iterations,
                                attempt=attempt, attempt_iterations=int(opt.nit),
                                NLL=value, optimizer_success=post_success,
                                optimizer_status=post_status, message=post_message,
                                gradient_inf_norm=float(np.max(np.abs(trial_gradient))),
                                accepted=accepted, maxls=line_search_budget))
            if post_success or post_status == 1:
                break
    ls, centers, lw = _unpack(theta, k, d)
    result = dict(sigma=float(np.exp(ls)), centers=centers.copy(), w=np.exp(lw),
                  U=_terms(X, centers, lw, m)[0], m=float(m), k=int(k), NLL=float(value),
                  theta=theta.copy(), history=history, converged=converged,
                  optimizer_failures=failures, post_success=post_success,
                  post_message=post_message, post_iterations=post_iterations,
                  post_status=post_status, post_attempts=post_attempts,
                  seed=int(seed), n_iter=iteration, engine='manuscript_defensive_rqmc',
                  fit_options=dict(max_outer=max_outer, epsilon=epsilon, delta=delta,
                                   inner_maxiter=inner_maxiter, K_post=K_post))
    result['IS_diagnostics'] = importance_diagnostics(result, Z, logq)
    return result

def sample_iid(fit, n, seed=123, max_proposals=10_000_000):
    """Independent rejection draws, with a global analytic Gaussian envelope.

    Let a_j = w_j ||x-v_j||^2 and C=k**(m-1). Then
    h(x) >= min_j(a_j)/C, hence exp(-h/sigma^2) <=
    sum_j exp(-w_j ||x-v_j||^2/(sigma^2*C)). Each summand is an
    unnormalized isotropic Gaussian. Select its component proportional to
    its integral and accept with exp(-h/sigma^2)/sum_j exp(-a_j/(sigma^2*C)).
    No fitted normalizing constant, burn-in, or MCMC approximation is needed.
    """
    if n < 1 or int(n) != n:
        raise ValueError('n must be a positive integer.')
    n = int(n)
    rng = np.random.default_rng(seed)
    centers, w = np.asarray(fit['centers']), np.asarray(fit['w'])
    sigma, m = fit['sigma'], fit['m']
    _pack(sigma, centers, w)
    if m <= 1 or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid fitted model.')
    k, d = centers.shape
    logvar = 2*np.log(sigma)+(m-1)*np.log(k)-np.log(w)-np.log(2.)
    logmass = .5*d*logvar
    component_prob = np.exp(logmass-logsumexp(logmass))
    sd = np.exp(.5*logvar)
    chunks, accepted, proposed = [], 0, 0
    while accepted < n:
        count = min(max(256, 2*(n-accepted)), 8192, max_proposals-proposed)
        if count <= 0:
            raise RuntimeError('Rejection sampler proposal budget exceeded; no shortened dataset is returned.')
        labels = rng.choice(k, size=count, p=component_prob)
        Y = centers[labels]+rng.normal(size=(count, d))*sd[labels, None]
        _, h, _ = _terms(Y, centers, np.log(w), m)
        d2 = cdist(Y, centers, 'sqeuclidean')
        log_envelope = logsumexp(-d2/(2*np.exp(logvar)), axis=1)
        log_accept = -h/sigma**2-log_envelope
        if np.max(log_accept) > 1e-8:
            raise RuntimeError('Numerical violation of rejection envelope.')
        keep = Y[np.log(rng.random(count)) <= np.minimum(log_accept, 0)]
        chunks.append(keep)
        accepted += len(keep)
        proposed += count
    return np.concatenate(chunks)[:n]

def align_parameters(centers, w, observed_centers):
    rows, cols = linear_sum_assignment(cdist(observed_centers, centers, 'sqeuclidean'))
    order = np.empty(len(rows), dtype=int)
    order[rows] = cols
    return np.asarray(centers)[order].copy(), np.asarray(w)[order].copy(), order
