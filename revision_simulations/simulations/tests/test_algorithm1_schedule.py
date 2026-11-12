import hashlib

import pytest

from simulations.algorithm1_experiments import atomic_json, configuration, initialize_run


def test_execution_parallelism_can_change_without_rewriting_scientific_config(tmp_path):
    from simulations.algorithm1_schedule import prepare_execution
    cfg = configuration('full', 4)
    initialize_run(tmp_path, cfg)
    path = tmp_path/'run_config.json'
    before = path.read_bytes()
    config, pending, execution = prepare_execution(tmp_path, jobs=8)
    assert path.read_bytes() == before
    assert config == cfg
    assert execution['jobs'] == 8 and execution['original_jobs'] == 4
    assert execution['run_config_sha256'] == hashlib.sha256(before).hexdigest()
    assert len(pending) == 2000


def test_resume_preserves_completed_record_and_skips_that_task(tmp_path):
    from simulations.algorithm1_schedule import prepare_execution
    cfg = configuration('full', 4)
    initialize_run(tmp_path, cfg)
    record = dict(experiment='estimation', n=200, replicate=0, success=False,
                  error='Recorded numerical failure, never a reason to replace the seed')
    path = tmp_path/'datasets'/'estimation_n200_r000.json.gz'
    atomic_json(path, record)
    before = path.read_bytes()
    _, pending, execution = prepare_execution(tmp_path, jobs=8)
    assert ('estimation', 200, 0) not in pending
    assert len(pending) == 1999
    assert path.read_bytes() == before
    assert execution['reused_sha256'][path.name] == hashlib.sha256(before).hexdigest()


def test_resume_rejects_changed_numerical_sources(tmp_path):
    from simulations.algorithm1_schedule import prepare_execution
    cfg = configuration('full', 4)
    cfg['source_sha256']['simulations/algorithm1_fit.py'] = 'different'
    initialize_run(tmp_path, cfg)
    with pytest.raises(ValueError, match='source'):
        prepare_execution(tmp_path, jobs=8)
