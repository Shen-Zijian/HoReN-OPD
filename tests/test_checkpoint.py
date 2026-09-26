"""Filesystem fault-injection and deterministic recovery tests (CPU only)."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from horen_opd.checkpoint import (
    atomic_json,
    canonical_hash,
    capture_rng,
    exclusive_run,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    sha256_file,
)


class AtomicJsonTests(unittest.TestCase):
    def test_roundtrip_unicode_and_stable_canonical_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested" / "state.json"
            payload = {"状态": "完成", "index": 12, "missing": None}
            atomic_json(target, payload)
            self.assertEqual(json.loads(target.read_text()), payload)
            self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))
            self.assertEqual(len(sha256_file(target)), 64)

    def test_nonfinite_or_replace_failure_preserves_prior_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state.json"
            atomic_json(target, {"valid": True})
            original = target.read_bytes()
            for invalid in [float("nan"), float("inf"), float("-inf")]:
                with self.assertRaises(ValueError):
                    atomic_json(target, {"value": invalid})
                self.assertEqual(target.read_bytes(), original)
            with mock.patch(
                "horen_opd.checkpoint.os.replace", side_effect=OSError("injected replace failure")
            ):
                with self.assertRaises(OSError):
                    atomic_json(target, {"valid": False})
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual([path.name for path in target.parent.iterdir()], ["state.json"])


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.rng_before = capture_rng()
        self.identity = {"config_sha256": "a" * 64, "data_sha256": "b" * 64, "source_sha256": "c" * 64}
        self.engine = {
            "adapter": {
                "values": torch.arange(6).reshape(2, 3),
                "key_labels": [torch.tensor([-1]), torch.tensor([7, 8])],
            }
        }
        self.run = {"edit_index": 1, "phase": "editing", "history": [{"case_id": 0}]}

    def tearDown(self):
        restore_rng(self.rng_before)
        self.temporary.cleanup()

    def save(self):
        return save_checkpoint(self.directory, self.engine, self.run, self.identity)

    def load(self):
        return load_checkpoint(self.directory / "latest.json", self.identity)

    def test_payload_and_pointer_complete_roundtrip(self):
        target = self.save()
        pointer = json.loads((self.directory / "latest.json").read_text())
        self.assertEqual(pointer["file"], target.name)
        self.assertEqual(pointer["sha256"], sha256_file(target))
        self.assertEqual(pointer["edit_index"], 1)
        loaded = self.load()
        self.assertEqual(loaded["identity"], self.identity)
        self.assertEqual(loaded["run"], self.run)
        self.assertTrue(torch.equal(loaded["engine"]["adapter"]["values"], self.engine["adapter"]["values"]))
        self.assertTrue(torch.equal(loaded["engine"]["adapter"]["key_labels"][1], torch.tensor([7, 8])))
        self.assertEqual(loaded["version"], 1)

    def test_sha_corruption_rejected_before_deserialization(self):
        target = self.save()
        with target.open("ab") as handle:
            handle.write(b"corrupt")
        with mock.patch("horen_opd.checkpoint.torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.load()
            loader.assert_not_called()

    def test_identity_mismatch_and_traversal_rejected(self):
        self.save()
        different = dict(self.identity, data_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "identity"):
            load_checkpoint(self.directory / "latest.json", different)
        record = json.loads((self.directory / "latest.json").read_text())
        record["file"] = "../outside.pt"
        atomic_json(self.directory / "latest.json", record)
        with self.assertRaisesRegex(ValueError, "sibling"):
            self.load()

    def test_rng_roundtrip_python_numpy_and_torch(self):
        random.seed(512)
        np.random.seed(512)
        torch.manual_seed(512)
        self.save()
        expected = (random.random(), np.random.random(4), torch.rand(4))
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restore_rng(self.load()["rng"])
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(4), expected[1])
        self.assertTrue(torch.equal(torch.rand(4), expected[2]))

    def test_cuda_rng_cannot_silently_resume_on_cpu(self):
        state = capture_rng()
        state["cuda"] = [torch.tensor([0], dtype=torch.uint8)]
        with mock.patch("horen_opd.checkpoint.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA RNG"):
                restore_rng(state)

    def test_latest_publish_failure_keeps_previous_checkpoint_valid(self):
        previous = self.save()
        pointer_before = (self.directory / "latest.json").read_bytes()
        self.run = dict(self.run, edit_index=2)
        self.engine = {"adapter": {"values": torch.zeros(3, 3)}}
        real_replace = os.replace

        def fail_pointer(source, destination):
            if Path(destination).name == "latest.json":
                raise OSError("injected latest-pointer publish failure")
            return real_replace(source, destination)

        with mock.patch("horen_opd.checkpoint.os.replace", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "publish failure"):
                self.save()
        self.assertEqual((self.directory / "latest.json").read_bytes(), pointer_before)
        self.assertTrue(previous.exists())
        self.assertEqual(self.load()["run"]["edit_index"], 1)
        # An orphan immutable payload is harmless. A later successful commit
        # publishes the next complete boundary and reclaims prior payloads.
        latest = self.save()
        self.assertEqual(self.load()["run"]["edit_index"], 2)
        self.assertEqual(list(self.directory.glob("state-*.pt")), [latest])

    def test_serialization_failure_keeps_previous_checkpoint_valid(self):
        self.save()
        before = (self.directory / "latest.json").read_bytes()
        with mock.patch("horen_opd.checkpoint.torch.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.save()
        self.assertEqual((self.directory / "latest.json").read_bytes(), before)
        self.assertEqual(self.load()["run"]["edit_index"], 1)
        self.assertEqual(list(self.directory.glob(".state-*.pt")), [])


@unittest.skipIf(sys.platform == "win32", "Advisory fcntl locking targets A100 Linux and local macOS")
class ExclusiveRunTests(unittest.TestCase):
    def test_excludes_second_owner_and_releases_after_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "inside"):
                with exclusive_run(temporary):
                    with self.assertRaisesRegex(RuntimeError, "already active"):
                        with exclusive_run(temporary):
                            self.fail("Second owner acquired lock")
                    raise ValueError("inside")
            with exclusive_run(temporary):
                pass

    def test_other_process_cannot_acquire_lock(self):
        code = "import fcntl,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)"
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / ".lock")
            with exclusive_run(temporary):
                blocked = subprocess.run(
                    [sys.executable, "-c", code, lock_path], capture_output=True, text=True, timeout=10
                )
            released = subprocess.run(
                [sys.executable, "-c", code, lock_path], capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("BlockingIOError", blocked.stderr)
            self.assertEqual(released.returncode, 0, released.stderr)
