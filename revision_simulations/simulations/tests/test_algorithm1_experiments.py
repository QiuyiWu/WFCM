import json

import numpy as np
import pytest


def _module():
    from simulations import algorithm1_experiments
    return algorithm1_experiments


def test_bootstrap_intervals_require_every_planned_replicate():
    mod = _module()
    draws = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    for method in ('normal', 'percentile'):
        with pytest.raises(ValueError, match='complete'):
            mod.bootstrap_intervals(draws, requested=5, method=method)
    lo, hi = mod.bootstrap_intervals(draws, requested=4, method='normal')
    np.testing.assert_allclose(lo, [0.3086533820502062, 3.086533820502062])
    np.testing.assert_allclose(hi, [4.691346617949794, 46.91346617949794])
    lo, hi = mod.bootstrap_intervals(draws, requested=4, method='percentile')
    np.testing.assert_allclose(lo, [1.075, 10.75])
    np.testing.assert_allclose(hi, [3.925, 39.25])


def test_incomplete_grid_never_selects_best_surviving_candidate():
    mod = _module()
    candidates = [dict(candidate_m=1.5, success=True, average_nll=4.),
                  dict(candidate_m=2., success=False, average_nll=3.)]
    assert mod.grid_choice(candidates, expected_m=(1.5, 2.)) is None
    candidates[1]['success'] = True
    assert mod.grid_choice(candidates, expected_m=(1.5, 2.)) == 2.


def test_absent_or_duplicate_grid_values_cannot_produce_a_selection():
    mod = _module()
    candidate = dict(candidate_m=2., success=True, average_nll=3.)
    assert mod.grid_choice([candidate], expected_m=(1.5, 2.)) is None
    assert mod.grid_choice([candidate, candidate], expected_m=(1.5, 2.)) is None


def test_resume_rejects_record_for_different_task_or_incomplete_grid(tmp_path):
    mod = _module()
    config = mod.configuration('smoke', 1)
    mod.initialize_run(tmp_path, config)
    target = tmp_path/'datasets'/'grid_n200_r000.json.gz'
    mod.atomic_json(target, dict(experiment='grid', n=500, replicate=0, success=True,
        candidates=[dict(candidate_m=2., success=True, average_nll=3.)], selected_m=2.))
    with pytest.raises(ValueError, match='record'):
        mod.run_task('grid', 200, 0, config, np.eye(4), tmp_path)
    record = mod.read_record(target)
    record['n'] = 200
    mod.atomic_json(target, record)
    with pytest.raises(ValueError, match='grid'):
        mod.run_task('grid', 200, 0, config, np.eye(4), tmp_path)


def test_resume_refuses_different_source_or_configuration(tmp_path):
    mod = _module()
    config = dict(seed=3, source_sha256={'engine': 'original'}, fit_options={'delta': 1e-5})
    mod.initialize_run(tmp_path, config)
    mod.initialize_run(tmp_path, config)
    for different in ({**config, 'seed': 4}, {**config, 'source_sha256': {'engine': 'changed'}}):
        with pytest.raises(ValueError, match='configuration'):
            mod.initialize_run(tmp_path, different)
    assert json.loads((tmp_path/'run_config.json').read_text()) == config


def test_failed_interval_remains_in_attempted_denominator():
    mod = _module()
    records = [dict(n=200, replicate=0, methods={
        'normal':dict(success=True, coverage=.5, length=2., cpu_seconds=1.)}),
        dict(n=200, replicate=1, methods={
        'normal':dict(success=False, coverage=None, length=None, cpu_seconds=3.)})]
    row = mod.coverage_summary(records).iloc[0]
    assert row.attempted_datasets == 2
    assert row.valid_datasets == 1
    assert row.coverage == .5
    assert row.success_and_cover == .25


def test_pilot_failure_blocks_expansion_even_if_another_method_passes():
    mod = _module()
    records = [dict(experiment='coverage', n=200, replicate=0,
                    methods={'fisher': {'success': True}, 'normal': {'success': False}})]
    gate = mod.pilot_gate(records)
    assert not gate['passed']
    assert gate['failed_tasks'] == 1


def test_full_run_rejects_successful_but_incomplete_pilot(tmp_path):
    mod = _module()
    config = mod.configuration('smoke', 1)
    mod.initialize_run(tmp_path, config)
    records = []
    for experiment in ('coverage', 'estimation', 'grid'):
        for n in config[experiment+'_ns']:
            for rep in range(config[experiment+'_reps']):
                record = dict(experiment=experiment, n=n, replicate=rep, success=True)
                if experiment == 'grid':
                    record['candidates'] = [dict(candidate_m=m, success=True, average_nll=m) for m in config['m_grid']]
                    record['selected_m'] = 1.5
                elif experiment == 'coverage':
                    record['methods'] = {'fisher': {'success': True}}
                    for method in ('normal', 'percentile'):
                        record['methods'][method] = dict(success=True,
                            bootstrap_requested=5, bootstrap_attempted=5, bootstrap_valid=5,
                            bootstrap_centers=[[1., 2.]]*5,
                            bootstrap_fits=[dict(index=b, success=True) for b in range(5)])
                mod.atomic_json(tmp_path/'datasets'/f'{experiment}_n{n}_r{rep:03d}.json.gz', record)
                records.append(record)
    mod.atomic_json(tmp_path/'pilot_gate.json', mod.pilot_gate(records))
    full = mod.configuration('full', 1)
    mod.validate_pilot(tmp_path, full)
    missing = tmp_path/'datasets'/'grid_n1000_r000.json.gz'
    missing.unlink()
    with pytest.raises(ValueError, match='complete'):
        mod.validate_pilot(tmp_path, full)


def test_full_run_rejects_a_forged_pass_for_partial_bootstrap(tmp_path):
    mod = _module()
    # The stored boolean is not sufficient proof of a complete pilot.
    mod.initialize_run(tmp_path, mod.configuration('smoke', 1))
    mod.atomic_json(tmp_path/'pilot_gate.json', {'passed': True})
    mod.atomic_json(tmp_path/'datasets'/'coverage_n200_r000.json.gz',
        dict(experiment='coverage', n=200, replicate=0, success=True,
             methods={'fisher': {'success': True}}))
    with pytest.raises(ValueError, match='complete'):
        mod.validate_pilot(tmp_path, mod.configuration('full', 1))
