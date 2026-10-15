"""Run a revision experiment using the original simulation drivers."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRIVERS = {
    "benchmark": ("benchmark_experiment", None),
    "computation": ("computation_experiment", "all"),
    "coverage": ("inference_experiment", "coverage"),
    "lrt": ("inference_experiment", "lrt"),
    "selection": ("inference_experiment", "selection"),
    "additional": ("legacy_corrected_experiment", "all"),
}


def confirmation(profile_name, output, jobs):
    """Reuse the archived confirmation engine and its independent seed schedule."""
    from joblib import Parallel, delayed
    from simulations import lrt_null_confirmation as engine

    identity = engine.validate_locked_engine(require_pinned_hash=True)
    profile = (engine.publication_profile() if profile_name == "full"
               else engine.smoke_profile())
    roots = engine.confirmation_root_schedule(profile.lrt_reps)
    config = dict(profile_name=profile_name, profile=asdict(profile), jobs=jobs,
                  seed=engine.CONFIRMATION_MASTER_SEED, engine=identity,
                  protocol=engine.CONFIRMATION_PROTOCOL_VERSION)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
        delayed(engine._confirmation_one)(profile, i, roots[i])
        for i in range(profile.lrt_reps)
    )
    frame = engine._write_rows(rows, output)
    engine.summarize_confirmation(frame, profile_name).to_csv(
        output / "lrt_null_confirmation_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*DRIVERS, "confirmation"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    output = (args.output or ROOT / "runs" / f"{args.experiment}_{args.profile}").resolve()
    archive = (ROOT / "results").resolve()
    if output == archive or archive in output.parents or output in archive.parents:
        parser.error("choose an output outside the archived results")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output is not empty; choose a new directory")
    # Keep timed comparisons on one worker and prevent nested BLAS threads.
    if args.experiment in {"benchmark", "computation", "coverage", "additional"} and args.jobs != 1:
        parser.error("use --jobs 1 for timed experiments")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))
    if args.experiment == "confirmation":
        confirmation(args.profile, output, args.jobs)
        return
    module, group = DRIVERS[args.experiment]
    seed = 20260906 if args.experiment == "computation" else 20260907
    command = [sys.executable, "-m", "simulations." + module]
    if group:
        command.append(group)
    command += ["--profile", args.profile, "--jobs", str(args.jobs),
                "--seed", str(seed), "--output", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
