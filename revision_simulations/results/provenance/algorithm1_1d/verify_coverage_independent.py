"""Read-only reconstruction of Table 6, without either WFCM fitter.

From revision_simulations:
PYTHONPATH=. python results/provenance/algorithm1_1d/verify_coverage_independent.py RUN OUTPUT

The m=2 energy and derivatives below are derived directly from a*b/(a+b),
independently of the fitting likelihood/gradient functions. Saved interval
endpoints and summaries are reconstructed from raw bootstrap centers.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from simulations.wfcm_core import WFCMParams, ISContext, build_proposal, sample_wfcm_1d


def seed(base, *indices):
    return int(np.random.SeedSequence([base, *indices]).generate_state(1, dtype=np.uint32)[0])


def closed_energy_and_gradient(x, sigma, centers, weights):
    x = np.asarray(x).reshape(-1)
    v1, v2 = np.asarray(centers).reshape(2)
    w1, w2 = weights
    a, b = w1*(x-v1)**2, w2*(x-v2)**2
    denominator = a+b
    if np.any(denominator <= 0):
        raise ValueError('Coincident centers at a sampled point require separate boundary analysis')
    energy = a*b/denominator/sigma**2
    gradient = np.column_stack([
        -2*energy,
        2*w1*(v1-x)*(b/denominator)**2/sigma**2,
        2*w2*(v2-x)*(a/denominator)**2/sigma**2,
        energy*(b*w2-a*w1)/denominator,
    ])
    return energy, gradient


def clean_fit(fit):
    return bool(fit['converged'] and not fit['optimizer_failures'] and
                (fit['post_success'] or fit['post_status'] == 1))


def fingerprint(values):
    values = np.asarray(values, dtype='<f8')
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype='<i8').tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def verify(run, output):
    run, output = Path(run), Path(output)
    config = json.loads((run/'run_config.json').read_text())
    for name, digest in config['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest() == digest
    records = [json.loads(gzip.decompress(p.read_bytes()))
               for p in sorted((run/'datasets').glob('coverage_*.json.gz'))]
    truth = WFCMParams(.9, np.array([[-2.], [2.]]), np.array([.35, .65]), 2.)
    expected = {(n, r) for n in (200, 500) for r in range(200)}
    actual = [(r['n'], r['replicate']) for r in records]
    assert len(set(actual)) == len(actual) and set(actual) <= expected
    rows, max_nll_error, max_ci_error, max_fisher_error = [], 0., 0., 0.
    requested_refits = attempted_refits = valid_refits = clipped_fisher = 0
    for record in records:
        n, rep = record['n'], record['replicate']
        observed = record.get('observed_fit')
        if not record['observed_error']:
            assert clean_fit(observed)
            params = WFCMParams(observed['sigma'], np.asarray(observed['centers']),
                               np.asarray(observed['w']), 2.)
            observed_centers = np.sort(params.centers[:, 0])
            np.testing.assert_array_equal(record['observed_centers'], observed_centers)
            data = sample_wfcm_1d(truth, n, seed(config['seed'], 1, n, rep))
            assert fingerprint(data) == record['data_sha256']
            proposal = build_proposal(data, 2, 'defensive', observed['proposal_seed'])
            context = ISContext.draw(proposal, max(4000, 20*n), observed['importance_seed'])
            e_data, _ = closed_energy_and_gradient(data, params.sigma, params.centers, params.weights)
            e_is, _ = closed_energy_and_gradient(context.samples, params.sigma, params.centers, params.weights)
            nll = e_data.sum()+n*(logsumexp(-e_is-context.log_q)-np.log(len(e_is)))
            error = abs(nll-observed['NLL'])
            max_nll_error = max(max_nll_error, error)
            assert error < 1e-7
        for method in ('normal', 'percentile', 'fisher'):
            item = record['methods'][method]
            if method != 'fisher':
                requested_refits += item['bootstrap_requested']
                attempted_refits += item['bootstrap_attempted']
                valid_refits += item['bootstrap_valid']
                assert item['bootstrap_requested'] == 199
                fits = item['bootstrap_fits']
                assert len(fits) == item['bootstrap_attempted']
                assert [b['index'] for b in fits] == list(range(len(fits)))
                valid = [b for b in fits if b['success']]
                assert len(valid) == item['bootstrap_valid']
                draws = np.asarray(item['bootstrap_centers']).reshape(-1, 2)
                assert len(draws) == len(valid)
                for b, values in zip(valid, draws):
                    assert clean_fit(b['fit'])
                    np.testing.assert_array_equal(values, np.sort(np.asarray(b['fit']['centers'])[:, 0]))
                if len(valid) != 199:
                    assert not item['success']
            if item['success']:
                assert not record['observed_error'], 'A valid interval requires a valid observed fit'
                if method == 'normal':
                    means = draws.sum(axis=0)/199
                    deviations = np.sqrt(((draws-means)**2).sum(axis=0)/199)
                    lower, upper = means-1.96*deviations, means+1.96*deviations
                elif method == 'percentile':
                    ordered = np.sort(draws, axis=0)
                    lower = .05*ordered[4]+.95*ordered[5]
                    upper = .95*ordered[193]+.05*ordered[194]
                else:
                    sample = sample_wfcm_1d(params, 50000, seed(config['seed'], 3, n, rep), grid_size=100001)
                    _, gradients = closed_energy_and_gradient(sample, params.sigma, params.centers, params.weights)
                    gradients -= gradients.mean(axis=0)
                    information = gradients.T@gradients/50000
                    information = (information+information.T)/2
                    eigenvalues, vectors = np.linalg.eigh(information)
                    floor = 1e-7*max(float(eigenvalues.max()), 1.)
                    clipped_fisher += int(np.any(eigenvalues < floor))
                    inverse = (vectors/np.maximum(eigenvalues, floor))@vectors.T
                    half = 1.959963984540054*np.sqrt(np.diag(inverse)[1:3]/n)
                    order = np.argsort(params.centers[:, 0])
                    lower, upper = (params.centers[:, 0]-half)[order], (params.centers[:, 0]+half)[order]
                ci_error = max(np.max(np.abs(lower-item['lower'])), np.max(np.abs(upper-item['upper'])))
                if method == 'fisher':
                    max_fisher_error = max(max_fisher_error, ci_error)
                    assert ci_error < 1e-8
                else:
                    max_ci_error = max(max_ci_error, ci_error)
                    assert ci_error < 1e-12
                covered = (lower <= [-2., 2.]) & (np.array([-2., 2.]) <= upper)
                coverage = float(covered.sum()/2)
                length = float((upper-lower).sum()/2)
                assert coverage == item['coverage']
                np.testing.assert_allclose(length, item['length'], atol=1e-8, rtol=0)
            else:
                coverage, length = np.nan, np.nan
                assert item['coverage'] is None and item['length'] is None
            assert np.isfinite(item['cpu_seconds']) and item['cpu_seconds'] >= 0
            rows.append(dict(n=n, replicate=rep, method=method, success=item['success'],
                             coverage=coverage, length=length, cpu_seconds=item['cpu_seconds']))
    summaries = []
    for (n, method), group in pd.DataFrame(rows).groupby(['n', 'method']):
        valid = group[group.success]
        result = dict(n=int(n), method=method, attempted=len(group), valid=len(valid),
                      success_and_cover=float(group.coverage.fillna(0.).sum()/len(group)))
        for metric in ('coverage', 'length', 'cpu_seconds'):
            array = valid[metric].to_numpy()
            result[metric] = float(array.mean()) if len(array) else None
            result[metric+'_mcse'] = float(array.std(ddof=1)/np.sqrt(len(array))) if len(array)>1 else None
        summaries.append(result)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output/'table6_independent_summary.csv', index=False)
    audit = dict(complete=set(actual)==expected, completed_datasets=len(records),
        observed_nll_checks=sum(not r['observed_error'] for r in records),
        bootstrap_requested=requested_refits, bootstrap_attempted=attempted_refits, bootstrap_valid=valid_refits,
        max_total_nll_error=float(max_nll_error), max_bootstrap_interval_error=float(max_ci_error),
        max_fisher_interval_error=float(max_fisher_error), fisher_eigenvalue_clipping_datasets=clipped_fisher,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'coverage_independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify(sys.argv[1], sys.argv[2])
