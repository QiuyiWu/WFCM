"""Deterministic, per-task checkpoints for long reviewer simulations.

The checkpoint sits outside every scientific worker and therefore outside the
worker's timed regions.  A completed task is serialized to a temporary file,
flushed to disk, and atomically renamed.  On restart, only complete files whose
task identity and payload digest both validate are reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np


CHECKPOINT_PROTOCOL = "atomic-independent-task-json-v1"
Task = TypeVar("Task", bound=tuple)


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_or_validate_config(path: Path, config: dict) -> None:
    """Write a new run config, or refuse to mix it with an existing run."""

    clean = _json_clean(config)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"existing run config is unreadable: {path}") from exc
        if existing != clean:
            raise RuntimeError(
                "output directory contains a different run configuration; "
                "use a new output directory rather than mixing checkpoints"
            )
        return
    _atomic_write(path, json.dumps(clean, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _checkpoint_path(directory: Path, index: int) -> Path:
    return directory / f"task_{index:06d}.json"


def _make_record(index: int, task: tuple, rows: Sequence[dict]) -> dict:
    body = {
        "protocol": CHECKPOINT_PROTOCOL,
        "task_index": int(index),
        "task": _json_clean(task),
        "rows": _json_clean(list(rows)),
    }
    return {
        **body,
        "payload_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }


def _validate_record(record: dict, index: int, task: tuple) -> list[dict]:
    required = {"protocol", "task_index", "task", "rows", "payload_sha256"}
    if set(record) != required:
        raise RuntimeError(f"checkpoint task {index} has an unexpected schema")
    body = {key: record[key] for key in required if key != "payload_sha256"}
    observed_digest = str(record["payload_sha256"])
    expected_digest = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    if observed_digest != expected_digest:
        raise RuntimeError(f"checkpoint task {index} failed its SHA-256 integrity check")
    if record["protocol"] != CHECKPOINT_PROTOCOL:
        raise RuntimeError(f"checkpoint task {index} uses an unsupported protocol")
    if record["task_index"] != index or record["task"] != _json_clean(task):
        raise RuntimeError(f"checkpoint task {index} has the wrong task identity")
    rows = record["rows"]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"checkpoint task {index} does not contain row records")
    return rows


def run_checkpointed_tasks(
    tasks: Iterable[Task],
    checkpoint_directory: Path,
    worker: Callable[[Task], Sequence[dict]],
    validate_rows: Callable[[Task, Sequence[dict]], None],
) -> list[list[dict]]:
    """Run or resume tasks in their declared order and return validated rows."""

    ordered = list(tasks)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    expected_names = {
        _checkpoint_path(checkpoint_directory, index).name
        for index in range(len(ordered))
    }
    unexpected = sorted(
        path.name
        for path in checkpoint_directory.glob("task_*.json")
        if path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(
            "checkpoint directory contains tasks outside the current schedule: "
            + ", ".join(unexpected[:5])
        )

    completed: list[list[dict]] = []
    reused = 0
    for index, task in enumerate(ordered):
        path = _checkpoint_path(checkpoint_directory, index)
        if path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"checkpoint task {index} is unreadable") from exc
            rows = _validate_record(record, index, task)
            reused += 1
        else:
            rows = list(worker(task))
            validate_rows(task, rows)
            record = _make_record(index, task, rows)
            _atomic_write(path, _canonical_bytes(record) + b"\n")
            # Reload the durable representation so fresh and resumed paths
            # pass through exactly the same validation and JSON conversion.
            rows = _validate_record(
                json.loads(path.read_text(encoding="utf-8")), index, task
            )
        validate_rows(task, rows)
        completed.append(rows)
        finished = index + 1
        if finished == len(ordered) or finished % max(1, len(ordered) // 100) == 0:
            print(
                f"checkpoint progress: {finished}/{len(ordered)} tasks "
                f"({reused} reused)",
                flush=True,
            )
    return completed


__all__ = [
    "CHECKPOINT_PROTOCOL",
    "run_checkpointed_tasks",
    "write_or_validate_config",
]
