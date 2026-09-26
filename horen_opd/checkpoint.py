"""Atomic, integrity-checked complete-edit boundaries (not metrics-only checkpoints)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import random
import tempfile

import numpy as np
import torch


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG checkpoint requires CUDA for reproducible resume")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(directory, engine_state, run_state, identity):
    """Commit immutable payload then atomically publish pointer; old pointer remains valid on crash."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".state-", suffix=".pt", dir=directory)
    os.close(fd)
    payload = dict(version=1, identity=identity, engine=engine_state, run=run_state, rng=capture_rng())
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        digest = sha256_file(temporary)
        target = directory / f"state-{digest}.pt"
        os.replace(temporary, target)
        atomic_json(
            directory / "latest.json",
            dict(
                file=target.name,
                sha256=digest,
                version=1,
                edit_index=run_state["edit_index"],
                phase=run_state["phase"],
                identity=identity,
            ),
        )

        for previous in directory.glob("state-*.pt"):
            if previous != target:
                previous.unlink()
        return target
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(pointer, expected_identity):
    pointer = Path(pointer)
    record = json.loads(pointer.read_text(encoding="utf-8"))
    name = record["file"]
    if Path(name).name != name:
        raise ValueError("Checkpoint pointer must reference a sibling file")
    target = pointer.parent / name
    if record["identity"] != expected_identity or sha256_file(target) != record["sha256"]:
        raise ValueError("Checkpoint identity or SHA256 mismatch")

    payload = torch.load(target, map_location="cpu", weights_only=False)
    if payload.get("version") != 1 or payload.get("identity") != expected_identity:
        raise ValueError("Invalid checkpoint payload")
    return payload


@contextmanager
def exclusive_run(directory):
    """Advisory OS lock; automatic release on process death, never stale PID polling."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run already active: {directory}") from exc
        yield
