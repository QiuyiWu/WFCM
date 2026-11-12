"""Change execution parallelism while preserving the frozen scientific run.

The original run_config and numerical source snapshots are never rewritten.
Each execution phase records its worker count and hashes reused checkpoints.
Stop the original driver before using this continuation scheduler.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from pathlib import Path
from time import perf_counter

from joblib import Parallel, delayed

from simulations.algorithm1_experiments import (
    ROOT, atomic_json, configuration, coverage_summary, pilot_gate, planned_tasks,
    read_record, run_task, truth_information, validate_pilot, validate_record,
)


def prepare_execution(output, jobs):
    output = Path(output)
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError('jobs must be a positive integer')
    config = read_record(output/'run_config.json')
    for name, digest in config['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Numerical source changed: {name}')
    if config != configuration(config['profile'], config['jobs']):
        raise ValueError('Frozen scientific configuration does not match the implemented protocol')
    expected = planned_tasks(config)
    pending, reused = [], {}
    for task in expected:
        experiment, n, rep = task
        path = output/'datasets'/f'{experiment}_n{n}_r{rep:03d}.json.gz'
        if path.exists():
            validate_record(read_record(path), task, config)
            reused[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            pending.append(task)
    execution = dict(started_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
        original_jobs=config['jobs'], numerical_threads_per_worker=1,
        run_config_sha256=hashlib.sha256((output/'run_config.json').read_bytes()).hexdigest(),
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_sha256=reused, pending_tasks=len(pending), status='running')
    return config, pending, execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--jobs', type=int, required=True)
    args = parser.parse_args()
    with (args.output/'continuation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another continuation scheduler owns this run')
        config, tasks, execution = prepare_execution(args.output, args.jobs)
        validate_pilot(args.pilot, config)
        phase = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        phase_path = args.output/'executions'/f'{phase}.json'
        atomic_json(phase_path, execution)
        inverse = truth_information(config, args.output)
        reused = len(execution['reused_sha256'])
        total = len(planned_tasks(config))
        print(f'{reused}/{total} saved tasks reused unchanged; {args.jobs} workers; '
              f'{len(tasks)} pending', flush=True)
        begin = perf_counter()
        try:
            completed = Parallel(n_jobs=args.jobs, return_as='generator_unordered')(
                delayed(run_task)(experiment, n, rep, config, inverse, args.output)
                for experiment, n, rep in tasks)
            for count, record in enumerate(completed, 1):
                print(f'{reused+count}/{total} {record["experiment"]} n={record["n"]} '
                    f'r={record["replicate"]} success={record["success"]}; '
                    f'phase elapsed {perf_counter()-begin:.1f}s', flush=True)
            records = [read_record(args.output/'datasets'/f'{kind}_n{n}_r{rep:03d}.json.gz')
                       for kind, n, rep in planned_tasks(config)]
            coverage_summary(records).to_csv(args.output/'coverage_summary.csv', index=False)
            atomic_json(args.output/'completion.json', pilot_gate(records))
            execution['status'] = 'complete'
        except BaseException as exc:
            execution.update(status='interrupted', reason=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            execution.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                             elapsed_seconds=perf_counter()-begin)
            atomic_json(phase_path, execution)


if __name__ == '__main__':
    main()
