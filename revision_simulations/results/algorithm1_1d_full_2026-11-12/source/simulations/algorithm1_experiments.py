"""Table 6 and one-dimensional additional studies using November fit_fixed.

Run the smoke profile before the full profile. Scientific n values and the
complete m grid are identical in both; only the replication counts differ.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter, process_time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from simulations.algorithm1_fit import FIT_OPTIONS, fit_algorithm1, fit_record
from simulations.inference_experiment import fisher_center_intervals
from simulations.manuscript_engine import align_parameters
from simulations.wfcm_core import (
    ISContext, WFCMParameterization, WFCMParams, build_proposal,
    energy_and_gradient_free, sample_wfcm_1d,
)


ROOT = Path(__file__).resolve().parents[1]
TRUTH = WFCMParams(.9, np.array([[-2.], [2.]]), np.array([.35, .65]), 2.)
M_GRID = (1.5, 1.7, 1.9, 2., 2.1, 2.3, 2.5)
COORDINATES = ('log_sigma', 'center_1', 'center_2', 'log_w1_over_w2')
SOURCE_FILES = (
    'simulations/algorithm1_fit.py', 'simulations/algorithm1_experiments.py',
    'simulations/manuscript_engine.py', 'simulations/wfcm_core.py',
    'simulations/inference_experiment.py',
)


def seed(base, *indices):
    return int(np.random.SeedSequence([int(base), *map(int, indices)]).generate_state(1)[0])


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), allow_nan=False, sort_keys=True).encode()
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == '.gz' else payload)
    temporary.replace(path)


def read_record(path):
    payload = Path(path).read_bytes()
    return json.loads(gzip.decompress(payload) if str(path).endswith('.gz') else payload)


def initialize_run(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'run_config.json'
    if path.exists() and read_record(path) != clean(config):
        raise ValueError('Run configuration or source changed; use a new output directory')
    atomic_json(path, config)
    (output/'datasets').mkdir(exist_ok=True)


def configuration(profile, jobs):
    full = profile == 'full'
    return dict(protocol='november-fit-fixed-1d-v1', profile=profile, seed=20260907,
        truth=dict(sigma=.9, centers=[[-2.], [2.]], weights=[.35, .65], m=2.),
        estimation_ns=[200, 500, 1000, 2000], estimation_reps=250 if full else 2,
        grid_ns=[200, 500, 1000], grid_reps=200 if full else 1, m_grid=list(M_GRID),
        coverage_ns=[200, 500], coverage_reps=200 if full else 2, B=199 if full else 5,
        observed_starts=3, bootstrap_starts=1, fit_options=FIT_OPTIONS,
        initialization='random observed centers, uniform weights, sigma=1; bootstrap observed-fit warm start',
        additional_importance='max(2000,ceil(n**1.5))', coverage_importance='max(4000,20*n)',
        proposal='unchanged wfcm_core defensive iid importance proposal',
        sampler='numerical inverse CDF, 200001 grid points, unchanged seeds',
        truth_fisher_draws=750000, fitted_fisher_draws=50000,
        bootstrap_normal_critical=1.96, bootstrap_normal_sd_ddof=0,
        failed_bootstrap_policy='no interval unless all B planned refits valid; no replacement',
        acceptance='block converged, no block optimizer failure, post_success or post_status==1',
        timing='CPU seconds including observed fit; one numerical thread per worker',
        jobs=jobs, python=platform.python_version(), numpy=np.__version__,
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES})


def bootstrap_intervals(draws, *, requested, method):
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or len(draws) != requested or requested < 2 or not np.isfinite(draws).all():
        raise ValueError('A complete finite set of planned bootstrap refits is required')
    if method == 'normal':
        center = draws.mean(axis=0)
        half = 1.96*draws.std(axis=0, ddof=0)
        return center-half, center+half
    if method == 'percentile':
        return tuple(np.quantile(draws, [.025, .975], axis=0))
    raise ValueError('Unknown bootstrap interval method')


def grid_choice(candidates, expected_m=M_GRID):
    if (sorted(c['candidate_m'] for c in candidates) != sorted(expected_m)
            or not all(c['success'] and np.isfinite(c['average_nll']) for c in candidates)):
        return None
    return min(candidates, key=lambda c: c['average_nll'])['candidate_m']


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for values in arrays:
        arr = np.asarray(values, dtype='<f8')
        digest.update(np.asarray(arr.shape, dtype='<i8').tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def truth_information(config, output):
    path = Path(output)/'truth_fisher.npz'
    if path.exists():
        with np.load(path) as data:
            return data['inverse'].copy()
    spec = WFCMParameterization(2, 1, m_fixed=2., weight_floor=0.)
    data = sample_wfcm_1d(TRUTH, config['truth_fisher_draws'], seed(config['seed'], 200))
    _, scores = energy_and_gradient_free(spec.pack(TRUTH), data, spec)
    scores -= scores.mean(axis=0)
    info = scores.T@scores/len(scores)
    info = (info+info.T)/2
    eigenvalues, vectors = np.linalg.eigh(info)
    clipped = np.maximum(eigenvalues, 1e-10*max(float(eigenvalues.max()), 1.))
    inverse = (vectors/clipped)@vectors.T
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, information=info, inverse=inverse, eigenvalues=eigenvalues,
                 truth_free=spec.pack(TRUTH), coordinates=COORDINATES)
    temporary.replace(path)
    return inverse


def estimation_one(n, rep, config, inverse):
    base = config['seed']
    data_seed, fit_seed = seed(base, 201, n, rep), seed(base, 202, n, rep)
    record = dict(experiment='estimation', n=n, replicate=rep, data_seed=data_seed,
                  fit_seed=fit_seed, success=False)
    try:
        x = sample_wfcm_1d(TRUTH, n, data_seed)
        fit = fit_algorithm1(x, 2, 2., importance_size=max(2000, int(np.ceil(n**1.5))),
                             seed=fit_seed, n_init=config['observed_starts'],
                             fit_options=config['fit_options'])
        centers, weights, _ = align_parameters(fit.params.centers, fit.params.weights, TRUTH.centers)
        aligned = WFCMParams(fit.params.sigma, centers, weights, 2.)
        spec = fit.parameterization
        difference = spec.pack(aligned)-spec.pack(TRUTH)
        record.update(success=fit.success, fit=fit_record(fit, history=True),
            data_sha256=fingerprint(x),
            importance_sha256=fingerprint(fit.importance_context.samples, fit.importance_context.log_q),
            coordinates=COORDINATES, estimate_free=spec.pack(aligned),
            truth_free=spec.pack(TRUTH), marginal_z=difference/np.sqrt(np.diag(inverse)/n),
            sigma_abs_error=abs(fit.params.sigma-TRUTH.sigma),
            center_rmse=float(np.sqrt(np.mean((centers-TRUTH.centers)**2))),
            weight_l1_error=float(np.abs(weights-TRUTH.weights).sum()),
            error='' if fit.success else 'Selected Algorithm 1 endpoint failed application checks')
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def grid_one(n, rep, config):
    base = config['seed']
    data_seed, fit_seed = seed(base, 211, n, rep), seed(base, 212, n, rep)
    record = dict(experiment='grid', n=n, replicate=rep, data_seed=data_seed,
                  fit_seed=fit_seed, success=False, candidates=[], selected_m=None)
    try:
        x = sample_wfcm_1d(TRUTH, n, data_seed)
        proposal = build_proposal(x, 2, 'defensive', seed(fit_seed, 1))
        context = ISContext.draw(proposal, max(2000, int(np.ceil(n**1.5))), seed(fit_seed, 2))
        record.update(data_sha256=fingerprint(x), importance_sha256=fingerprint(context.samples, context.log_q))
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        return record
    for m in config['m_grid']:
        item = dict(candidate_m=m, success=False, average_nll=None)
        try:
            fit = fit_algorithm1(x, 2, m, context=context, seed=seed(fit_seed, 3),
                n_init=config['observed_starts'], fit_options=config['fit_options'])
            item.update(success=fit.success, average_nll=fit.average_nll,
                        fit=fit_record(fit, history=True))
        except Exception as exc:
            item['error'] = f'{type(exc).__name__}: {exc}'
        record['candidates'].append(item)
    record['selected_m'] = grid_choice(record['candidates'], config['m_grid'])
    record['success'] = record['selected_m'] is not None
    if record['success']:
        minimum = min(c['average_nll'] for c in record['candidates'])
        for item in record['candidates']:
            item['excess_average_nll'] = item['average_nll']-minimum
    return record


def coverage_one(n, rep, config):
    base = config['seed']
    data_seed, fit_seed = seed(base, 1, n, rep), seed(base, 2, n, rep)
    record = dict(experiment='coverage', n=n, replicate=rep, data_seed=data_seed,
                  fit_seed=fit_seed, success=False, methods={})
    x = sample_wfcm_1d(TRUTH, n, data_seed)
    begin = process_time()
    fit, error = None, ''
    try:
        fit = fit_algorithm1(x, 2, 2., importance_size=max(4000, 20*n), seed=fit_seed,
            n_init=config['observed_starts'], fit_options=config['fit_options'])
        record['observed_fit'] = fit_record(fit, history=True)
        if not fit.success:
            raise RuntimeError('Observed Algorithm 1 fit failed application checks')
        centers, _, order = align_parameters(fit.params.centers, fit.params.weights, TRUTH.centers)
        record.update(observed_centers=centers[:, 0], data_sha256=fingerprint(x))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    base_cpu = process_time()-begin
    record['observed_error'] = error
    for method in ('fisher', 'normal', 'percentile'):
        start = process_time()
        item = dict(success=False, coverage=None, length=None, error=error)
        if method != 'fisher':
            item.update(bootstrap_requested=config['B'], bootstrap_attempted=0,
                        bootstrap_valid=0, bootstrap_fits=[], bootstrap_centers=[])
        if not error:
            try:
                if method == 'fisher':
                    lo, hi, diagnostics = fisher_center_intervals(fit, n,
                        draws=config['fitted_fisher_draws'], seed=seed(base, 3, n, rep))
                    lo, hi = lo[order], hi[order]
                    item['fisher_diagnostics'] = diagnostics
                else:
                    boot_seed = seed(base, 4, n, rep)
                    rng = np.random.default_rng(boot_seed)
                    draws = []
                    for b in range(config['B']):
                        refit_seed = seed(boot_seed, b, 771)
                        boot = dict(index=b, seed=refit_seed, success=False)
                        item['bootstrap_attempted'] += 1
                        try:
                            if method == 'normal':
                                sample_seed = seed(boot_seed, b, 881)
                                sample = sample_wfcm_1d(fit.params, n, sample_seed)
                                boot['sample_seed'] = sample_seed
                            else:
                                sample = x[rng.integers(0, n, size=n)]
                            refit = fit_algorithm1(sample, 2, 2., importance_size=max(4000, 20*n),
                                seed=refit_seed, n_init=1, initial_params=fit.params,
                                fit_options=config['fit_options'])
                            boot.update(fit=fit_record(refit), success=refit.success)
                            if not refit.success:
                                raise RuntimeError('Bootstrap Algorithm 1 fit failed application checks')
                            aligned, _, _ = align_parameters(refit.params.centers,
                                refit.params.weights, fit.params.centers)
                            draws.append(aligned[order, 0])
                        except Exception as exc:
                            boot.update(success=False, error=f'{type(exc).__name__}: {exc}')
                        item['bootstrap_fits'].append(boot)
                    item.update(bootstrap_valid=len(draws), bootstrap_centers=draws)
                    lo, hi = bootstrap_intervals(np.asarray(draws).reshape(-1, 2),
                        requested=config['B'], method=method)
                if not (np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi >= lo)):
                    raise ValueError('Nonfinite or reversed interval endpoints')
                covered = (lo <= TRUTH.centers[:, 0]) & (TRUTH.centers[:, 0] <= hi)
                item.update(success=True, lower=lo, upper=hi, coverage=float(covered.mean()),
                            covered=covered, length=float(np.mean(hi-lo)), error='')
            except Exception as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
        item['cpu_seconds'] = base_cpu+process_time()-start
        record['methods'][method] = item
    record['success'] = all(item['success'] for item in record['methods'].values())
    return record


def run_task(experiment, n, rep, config, inverse, output):
    path = Path(output)/'datasets'/f'{experiment}_n{n}_r{rep:03d}.json.gz'
    if path.exists():
        record = read_record(path)
        validate_record(record, (experiment, n, rep), config)
        return record
    with threadpool_limits(limits=1):
        if experiment == 'estimation':
            record = estimation_one(n, rep, config, inverse)
        elif experiment == 'grid':
            record = grid_one(n, rep, config)
        else:
            record = coverage_one(n, rep, config)
    validate_record(record, (experiment, n, rep), config)
    atomic_json(path, record)
    return clean(record)


def coverage_summary(records):
    rows = [dict(n=r['n'], replicate=r['replicate'], method=method, **
                 {key: item[key] for key in ('success', 'coverage', 'length', 'cpu_seconds')})
            for r in records for method, item in r.get('methods', {}).items()]
    if not rows:
        return pd.DataFrame()
    results = []
    for (n, method), group in pd.DataFrame(rows).groupby(['n', 'method']):
        valid = group[group.success]
        row = dict(n=n, method=method, attempted_datasets=len(group), valid_datasets=len(valid),
                   failed_datasets=len(group)-len(valid),
                   success_and_cover=group.coverage.where(group.success, 0.).mean())
        for metric in ('coverage', 'length', 'cpu_seconds'):
            values = valid[metric].astype(float)
            row[metric] = values.mean()
            row[metric+'_mcse'] = values.std(ddof=1)/np.sqrt(len(values)) if len(values) else np.nan
        results.append(row)
    return pd.DataFrame(results)


def pilot_gate(records):
    failed = []
    for r in records:
        ok = (all(m['success'] for m in r['methods'].values()) if r['experiment'] == 'coverage'
              else r['success'])
        if not ok:
            failed.append(dict(experiment=r['experiment'], n=r['n'], replicate=r['replicate']))
    return dict(passed=bool(records) and not failed, attempted_tasks=len(records),
                failed_tasks=len(failed), failures=failed)


def validate_record(record, task, config):
    if tuple(record.get(k) for k in ('experiment', 'n', 'replicate')) != tuple(task):
        raise ValueError('Saved record identity does not match the planned task')
    if record['experiment'] == 'grid':
        candidates = record.get('candidates', [])
        if not candidates and not record['success'] and record.get('error'):
            return  # An explicitly recorded proposal/data setup failure.
        if sorted(c['candidate_m'] for c in candidates) != sorted(config['m_grid']):
            raise ValueError('Saved grid record is missing or duplicating prescribed candidates')
        selected = grid_choice(candidates, config['m_grid'])
        if selected != record.get('selected_m') or record['success'] != (selected is not None):
            raise ValueError('Saved grid selection contradicts candidate validity or NLL')
    elif record['experiment'] == 'coverage':
        methods = record.get('methods', {})
        if set(methods) != {'fisher', 'normal', 'percentile'}:
            raise ValueError('Saved coverage record has an incomplete method set')
        for name in ('normal', 'percentile'):
            item = methods[name]
            count = config['B']
            if item['bootstrap_requested'] != count:
                raise ValueError('Saved bootstrap record has a different requested count')
            fits = item['bootstrap_fits']
            if (len(fits) != item['bootstrap_attempted'] or
                    [f['index'] for f in fits] != list(range(len(fits))) or
                    sum(f['success'] for f in fits) != item['bootstrap_valid'] or
                    len(item['bootstrap_centers']) != item['bootstrap_valid']):
                raise ValueError('Saved bootstrap record has inconsistent attempted/valid counts')
            if item['success'] and (item['bootstrap_valid'] != count or
                    not np.isfinite(item['bootstrap_centers']).all()):
                raise ValueError('A successful interval needs complete finite bootstrap estimates')
        if record['success'] != all(m['success'] for m in methods.values()):
            raise ValueError('Saved coverage record has inconsistent success status')


def planned_tasks(config, experiment='all'):
    return [(kind, n, rep) for kind in ('coverage', 'estimation', 'grid')
            if experiment in ('all', kind) for n in config[kind+'_ns']
            for rep in range(config[kind+'_reps'])]


def validate_pilot(output, full_config):
    output = Path(output)
    config = read_record(output/'run_config.json')
    expected_config = configuration('smoke', config['jobs'])
    if config != expected_config or config['source_sha256'] != full_config['source_sha256']:
        raise ValueError('Pilot configuration and sources must match the complete prescribed smoke profile')
    records = [read_record(p) for p in sorted((output/'datasets').glob('*.json.gz'))]
    identities = [(r['experiment'], r['n'], r['replicate']) for r in records]
    if sorted(identities) != sorted(planned_tasks(config)):
        raise ValueError('A complete pilot must contain every prescribed task exactly once')
    for record, identity in zip(records, identities):
        validate_record(record, identity, config)
    gate = pilot_gate(records)
    if not gate['passed']:
        raise ValueError('The complete pilot contains numerical failures')
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--experiment', choices=['all', 'coverage', 'estimation', 'grid'], default='all')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, help='Successful complete smoke run required for full execution')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be positive')
    config = configuration(args.profile, args.jobs)
    if args.profile == 'full':
        if args.pilot is None:
            parser.error('Run and supply a successful smoke pilot first')
        try:
            validate_pilot(args.pilot, config)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    initialize_run(args.output, config)
    snapshot = args.output/'source'
    for name in SOURCE_FILES:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    inverse = truth_information(config, args.output)
    tasks = planned_tasks(config, args.experiment)
    print(f'Protocol {config["protocol"]}: {len(tasks)} planned tasks', flush=True)
    start = perf_counter()
    completed = Parallel(n_jobs=args.jobs, return_as='generator_unordered')(
        delayed(run_task)(experiment, n, rep, config, inverse, args.output)
        for experiment, n, rep in tasks)
    records = []
    for record in completed:
        records.append(record)
        print(f'{len(records)}/{len(tasks)} {record["experiment"]} n={record["n"]} '
              f'r={record["replicate"]} success={record["success"]}; {perf_counter()-start:.1f}s', flush=True)
    coverage_summary(records).to_csv(args.output/'coverage_summary.csv', index=False)
    gate = pilot_gate(records)
    atomic_json(args.output/('pilot_gate.json' if args.profile == 'smoke' else 'completion.json'), gate)
    print(json.dumps(gate), flush=True)


if __name__ == '__main__':
    main()
