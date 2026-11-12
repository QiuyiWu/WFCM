"""Reconstruct Figures 10--12 summaries and rescore every saved endpoint.

No optimizer is called. Run from revision_simulations with PYTHONPATH=.:
python results/provenance/algorithm1_1d/verify_additional_independent.py RUN SUMMARY OUTPUT
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
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from simulations.wfcm_core import WFCMParams, ISContext, build_proposal, sample_wfcm_1d
from verify_coverage_independent import seed, closed_energy_and_gradient, clean_fit, fingerprint


def cloud_fingerprint(context):
    digest = hashlib.sha256()
    for array in (context.samples, context.log_q):
        values = np.asarray(array, dtype='<f8')
        digest.update(np.asarray(values.shape, dtype='<i8').tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def energy(x, fit):
    distances = np.maximum((np.asarray(x).reshape(-1, 1)-np.asarray(fit['centers']).reshape(1, 2))**2, 1e-200)
    log_terms = (np.log(distances)+np.log(fit['w']))/(1-fit['m'])
    return np.exp((1-fit['m'])*np.logaddexp(log_terms[:, 0], log_terms[:, 1]))/fit['sigma']**2


def rescore(data, context, fit):
    value = energy(data, fit).sum()+len(data)*(logsumexp(-energy(context.samples, fit)-context.log_q)
                                            -np.log(len(context.samples)))
    error = abs(value-fit['NLL'])
    assert error < 1e-7
    np.testing.assert_allclose(fit['NLL']/len(data), fit['average_nll'], rtol=0, atol=1e-12)
    assert fit['NLL'] == min(s['NLL'] for s in fit['starts'])
    assert fit['success'] == clean_fit(fit)
    return float(error)


def verify(run, summary, output):
    run, summary, output = Path(run), Path(summary), Path(output)
    config = json.loads((run/'run_config.json').read_text())
    assert config['profile'] == 'full'
    for name, digest in config['source_sha256'].items():
        assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest() == digest
    records = [json.loads(gzip.decompress(p.read_bytes()))
               for pattern in ('estimation_*.json.gz', 'grid_*.json.gz')
               for p in sorted((run/'datasets').glob(pattern))]
    actual = [(r['experiment'], r['n'], r['replicate']) for r in records]
    expected = {('estimation', n, r) for n in (200, 500, 1000, 2000) for r in range(250)}
    expected |= {('grid', n, r) for n in (200, 500, 1000) for r in range(200)}
    assert len(set(actual)) == len(actual) and set(actual) == expected
    truth = WFCMParams(.9, np.array([[-2.], [2.]]), np.array([.35, .65]), 2.)
    truth_free = np.array([np.log(.9), -2., 2., np.log(.35/.65)])
    sample = sample_wfcm_1d(truth, 750000, seed(config['seed'], 200))
    _, scores = closed_energy_and_gradient(sample, truth.sigma, truth.centers, truth.weights)
    scores -= scores.mean(axis=0)
    information = scores.T@scores/len(scores)
    information = (information+information.T)/2
    eigenvalues, vectors = np.linalg.eigh(information)
    floor = 1e-10*max(float(eigenvalues.max()), 1.)
    inverse = (vectors/np.maximum(eigenvalues, floor))@vectors.T
    with np.load(run/'truth_fisher.npz') as saved:
        np.testing.assert_allclose(information, saved['information'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(inverse, saved['inverse'], atol=1e-10, rtol=0)
    estimate_rows, normality_rows, curve_rows, selection_rows = [], [], [], []
    endpoint_checks, max_nll_error, raw_coordinate_corrections = 0, 0., 0
    reconstructed, extremes = [], []
    for r in records:
        n, rep, kind = r['n'], r['replicate'], r['experiment']
        m_size = max(2000, int(np.ceil(n**1.5)))
        data_seed = seed(config['seed'], 201 if kind == 'estimation' else 211, n, rep)
        assert r['data_seed'] == data_seed
        data = sample_wfcm_1d(truth, n, data_seed)
        if 'data_sha256' in r:
            assert r['data_sha256'] == fingerprint(data)
        if kind == 'estimation':
            if 'fit' not in r:
                assert not r['success'] and r['error']
                continue
            fit = r['fit']
            proposal = build_proposal(data, 2, 'defensive', fit['proposal_seed'])
            context = ISContext.draw(proposal, m_size, fit['importance_seed'])
            assert r['importance_sha256'] == cloud_fingerprint(context)
            max_nll_error = max(max_nll_error, rescore(data, context, fit))
            endpoint_checks += 1
            assert r['success'] == clean_fit(fit)
            order = np.argsort(np.asarray(fit['centers'])[:, 0])
            centers, weights = np.asarray(fit['centers'])[order, 0], np.asarray(fit['w'])[order]
            free = np.r_[np.log(fit['sigma']), centers, np.log(weights[0]/weights[1])]
            # The frozen driver's generic packer clipped weights at 1e-12.
            # Verify that original diagnostic, then independently reconstruct
            # the exact physical log-ratio used by the published QQ plots.
            archived_free = free.copy()
            clipped = np.maximum(weights, 1e-12)
            archived_free[3] = np.log(clipped[0])-np.log(clipped[1])
            np.testing.assert_allclose(archived_free, r['estimate_free'], atol=1e-12, rtol=0)
            raw_coordinate_corrections += int(not np.allclose(free, archived_free, atol=1e-10, rtol=0))
            np.testing.assert_allclose(truth_free, r['truth_free'], atol=1e-12, rtol=0)
            z = (free-truth_free)/np.sqrt(np.diag(inverse)/n)
            archived_z = (archived_free-truth_free)/np.sqrt(np.diag(inverse)/n)
            np.testing.assert_allclose(archived_z, r['marginal_z'], atol=1e-9, rtol=0)
            if max(abs(z)) > 5:
                reference = dict(sigma=.9, centers=[[-2.], [2.]], w=[.35, .65], m=2.)
                reference_nll = energy(data, reference).sum()+n*(
                    logsumexp(-energy(context.samples, reference)-context.log_q)-np.log(m_size))
                extremes.append(dict(experiment=kind, n=n, replicate=rep, sigma=fit['sigma'],
                    centers=fit['centers'], weights=fit['w'], marginal_z=z.tolist(), NLL=fit['NLL'],
                    true_parameter_NLL_same_cloud=float(reference_nll),
                    objective_gap_to_feasible_truth=float(fit['NLL']-reference_nll)))
            metrics = dict(center_rmse=np.sqrt(np.mean((centers-[-2., 2.])**2)),
                           sigma_abs_error=abs(fit['sigma']-.9),
                           weight_l1_error=np.abs(weights-[.35, .65]).sum())
            for key, value in metrics.items():
                np.testing.assert_allclose(value, r[key], atol=1e-12, rtol=0)
            if r['success']:
                reconstructed.append(dict(n=n, z=z, **metrics))
        else:
            candidates = r['candidates']
            if not candidates:
                assert not r['success'] and r['error']
                continue
            assert [c['candidate_m'] for c in candidates] == [1.5, 1.7, 1.9, 2., 2.1, 2.3, 2.5]
            fit_seed = seed(config['seed'], 212, n, rep)
            proposal = build_proposal(data, 2, 'defensive', seed(fit_seed, 1))
            context = ISContext.draw(proposal, m_size, seed(fit_seed, 2))
            assert r['importance_sha256'] == cloud_fingerprint(context)
            for c in candidates:
                if 'fit' in c:
                    max_nll_error = max(max_nll_error, rescore(data, context, c['fit']))
                    endpoint_checks += 1
                    assert c['fit']['m'] == c['candidate_m']
                    assert c['success'] == clean_fit(c['fit'])
                    np.testing.assert_allclose(c['average_nll'], c['fit']['NLL']/n, atol=1e-12, rtol=0)
                else:
                    assert not c['success'] and c['error']
            admissible = all(c['success'] for c in candidates)
            best = min(candidates, key=lambda c: c['average_nll'])['candidate_m'] if admissible else None
            assert r['success'] == admissible and r['selected_m'] == best

    for n in (200, 500, 1000, 2000):
        valid = [r for r in reconstructed if r['n'] == n]
        row = dict(n=n, attempted=250, valid=len(valid))
        for metric in ('center_rmse', 'sigma_abs_error', 'weight_l1_error'):
            a = np.array([r[metric] for r in valid])
            row.update({metric: a.mean(), metric+'_sd': a.std(ddof=1), metric+'_mcse': a.std(ddof=1)/np.sqrt(len(a))})
        estimate_rows.append(row)
        for j, coordinate in enumerate(('log_sigma', 'center_1', 'center_2', 'log_w1_over_w2')):
            z = np.sort([r['z'][j] for r in valid])
            normality_rows.append(dict(n=n, coordinate=coordinate, attempted=250, valid=len(z),
                mean=z.mean(), sd=z.std(ddof=1), q025=np.quantile(z, .025), q975=np.quantile(z, .975),
                qq_rmse=np.sqrt(np.mean((z-norm.ppf((np.arange(len(z))+.5)/len(z)))**2))))
    for n in (200, 500, 1000):
        valid = [r for r in records if r['experiment']=='grid' and r['n']==n and r['success']]
        for m in (1.5, 1.7, 1.9, 2., 2.1, 2.3, 2.5):
            values = np.array([next(c['fit']['NLL'] for c in r['candidates'] if c['candidate_m']==m) for r in valid])
            curve_rows.append(dict(n=n, m=m, attempted=200, valid=len(valid), mean_total_nll=values.mean(),
                sd_total_nll=values.std(ddof=1), mcse_total_nll=values.std(ddof=1)/np.sqrt(len(values)),
                mean_average_nll=values.mean()/n, sd_average_nll=values.std(ddof=1)/n,
                mcse_average_nll=values.std(ddof=1)/np.sqrt(len(values))/n))
            count = sum(r['selected_m']==m for r in valid)
            selection_rows.append(dict(n=n, m=m, count=count, attempted=200, valid=len(valid),
                                       frequency_valid=count/len(valid), frequency_attempted=count/200))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('estimation', estimate_rows), ('normality', normality_rows),
                       ('m_grid', curve_rows), ('m_selection', selection_rows)):
        independent = pd.DataFrame(rows)
        reported = pd.read_csv(summary/f'{name}.csv')
        pd.testing.assert_frame_equal(independent[reported.columns], reported, check_dtype=False,
                                      check_exact=False, atol=1e-9, rtol=1e-10)
        independent.to_csv(output/f'{name}_independent.csv', index=False)
    audit = dict(complete=True, task_records=len(records), endpoint_checks=endpoint_checks,
                 max_total_nll_error=max_nll_error, summary_tables_verified=4,
                 raw_coordinate_corrections=raw_coordinate_corrections,
                 truth_fisher_clipped=bool(np.any(eigenvalues < floor)),
                 verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'estimation_extremes.json').write_text(json.dumps(dict(
        diagnostic='Descriptive only: any absolute marginal z > 5; no exclusion or refitting',
        records=extremes), indent=2)+'\n')
    (output/'additional_independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        verify(*sys.argv[1:])
