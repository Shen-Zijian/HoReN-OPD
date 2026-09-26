import copy
from contextlib import contextmanager
import json

import pytest

from horen_opd.checkpoint import capture_rng, restore_rng
from horen_opd.config import RunConfig
from horen_opd.run import preservation_prompts, run_stream
from horen_opd.score import score_unke, score_zsre
from test_engine import assert_nested_equal, backbone, make_engine
from test_zsre_metrics import FakeEngine, request


class StreamEngine(FakeEngine):
    def __init__(self):
        super().__init__({"edit": "cat", "paraphrase": "cat", "neighbor": "dog"})
        self.edited = []
        self.preservation = []
        self.fail_at = None

    def edit(self, item, prompts, preserve):
        if item["case_id"] == self.fail_at:
            raise RuntimeError("injected interruption")
        self.edited.append(item["case_id"])
        self.preservation.append((preserve, prompts))
        return {"edit_index": len(self.edited)}

    def snapshot(self):
        return {"edited": list(self.edited), "preservation": copy.deepcopy(self.preservation)}

    def restore(self, state):
        self.edited, self.preservation = state["edited"], state["preservation"]

    def stats(self):
        return {"num_memory_slots": len(self.edited)}

    @contextmanager
    def original_context(self):
        yield


@pytest.mark.parametrize("method", ["baseline", "preserve"])
def test_complete_stream_and_resume_match(tmp_path, method):
    cfg = RunConfig(n=3, method=method, eval_every=1)
    edits = [request(i) for i in range(3)]
    pool = {"train": [{"task": "gsm8k", "prompt": "Train only"}]}
    expected = StreamEngine()
    full = tmp_path / "full"
    full.mkdir()
    a = run_stream(cfg, expected, edits, pool, full, "same")
    interrupted = StreamEngine()
    interrupted.fail_at = 1
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(RuntimeError, match="interruption"):
        run_stream(cfg, interrupted, edits, pool, partial, "same")
    resumed = StreamEngine()
    b = run_stream(cfg, resumed, edits, pool, partial, "same", resume=True)
    assert a["metrics"] == b["metrics"] == {"Rel": 1, "Gen": 1, "Loc": 1}
    assert resumed.edited == [0, 1, 2]
    assert resumed.preservation == expected.preservation
    assert b["evaluation_checkpoints"] == [1, 2, 3]
    evaluation = json.loads((partial / "evaluations/0003.json").read_text())
    assert score_zsre(evaluation) == b["metrics"]


def test_pending_evaluation_resumes_without_reediting(tmp_path, monkeypatch):
    import horen_opd.run as runner

    cfg = RunConfig(n=2, eval_every=1)
    engine = StreamEngine()
    real = runner.evaluate_edits
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("evaluation interrupted")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_edits", interrupt)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        run_stream(cfg, engine, [request(0), request(1)], {"train": []}, tmp_path, "id")
    monkeypatch.setattr(runner, "evaluate_edits", real)
    recovered = StreamEngine()
    run_stream(cfg, recovered, [request(0), request(1)], {"train": []}, tmp_path, "id", resume=True)
    assert recovered.edited == [0, 1]


def test_opd_freezes_backbone_teacher_and_restores_sampling(tmp_path):
    engine = make_engine()
    frozen = backbone(engine)
    teachers = []
    compute = engine._kl_on_fresh_rollout

    def capture(prompt, teacher):
        saved = copy.deepcopy(teacher)
        result = compute(prompt, teacher)
        assert_nested_equal(teacher, saved)
        teachers.append(saved)
        return result

    engine._kl_on_fresh_rollout = capture
    engine.edit({"prompt": "xy", "target_new": "z"}, ["ab"], preserve=True)
    assert len(teachers) == 1
    assert_nested_equal(frozen, backbone(engine))
    state, rng = engine.snapshot(), capture_rng()
    log = engine.edit({"prompt": "cd", "target_new": "e"}, ["xy"], preserve=True)
    expected = engine.snapshot()
    engine.restore(state)
    restore_rng(rng)
    replay = engine.edit({"prompt": "cd", "target_new": "e"}, ["xy"], preserve=True)
    assert log["losses"] == replay["losses"]
    assert log["opd"] == replay["opd"]
    assert_nested_equal(engine.snapshot(), expected)
    assert all(not p.requires_grad for p in engine.model.parameters())
    assert not hasattr(engine.adapter, "global_lora_A")


def test_unke_schedule_and_no_fabricated_loc(tmp_path, monkeypatch):
    import horen_opd.evaluation as evaluation
    from test_unke_metrics import Engine, request as unke_request

    cfg = RunConfig(dataset="unke", n=2, similarity_model_path="local")
    assert RunConfig(dataset="unke", n=1000).evaluation_checkpoints == [1, 10, 30, 100, 120, 500, 1000]
    engine = StreamEngine()
    engine.tokenizer = Engine.tokenizer
    engine.outputs = {
        f"chat-{name}-{i}": "the target answer" for name in ("original", "para") for i in (0, 1)
    }
    monkeypatch.setattr(
        evaluation, "_sentence_embedding_similarity", lambda refs, preds, *a: [0.5] * len(refs)
    )
    result = run_stream(cfg, engine, [unke_request(0), unke_request(1)], {"train": []}, tmp_path, "id")
    assert result["metrics"]["Loc"] is None
    assert json.loads((tmp_path / "original.json").read_text()) == {"status": "not_evaluated"}
    saved = json.loads((tmp_path / "evaluations/0002.json").read_text())
    assert score_unke(saved, "local", lambda refs, preds: [0.5] * len(refs)) == result["metrics"]


def test_replay_uses_only_seen_original_prompts():
    pool = {"train": [{"task": "gsm8k", "prompt": "reasoning"}]}
    edits = [
        {"case_id": i, "prompt": f"edit{i}", "target_new": "target", "rephrase_prompt": f"eval{i}"}
        for i in range(3)
    ]
    import random

    prompts = preservation_prompts(edits, 1, pool, random.Random(42))
    assert prompts[0] == "edit0"
    assert len(prompts) == 2
    assert all("edit2" not in p and "eval" not in p for p in prompts)


def test_invalid_config_rejected():
    for values in (
        {"method": "preserve_collapse"},
        {"n": 0},
        {"opd_steps": 0},
        {"dataset": "unke"},
        {"method": "preserve", "reasoning_path": None},
    ):
        with pytest.raises(ValueError):
            RunConfig(**values).validate()


def test_mquake_stream_and_prediction_rescore(tmp_path, monkeypatch):
    from horen_opd.mquake import build_protocol, summarize
    from test_mquake import row

    plan = build_protocol([row(1, ["train_edited"]), row(2, ["test_unedited"])], 100)
    engine = StreamEngine()
    engine.outputs["Someone works at"] = "NEW"
    calls = []

    def generate(engine, question, budget):
        calls.append((question, budget))
        answer = {"Q1a": "wrong", "Q1b": "new alias", "Q2a": "OLD"}[question]
        return answer, {"raw": answer, "tokens": 1, "elapsed_seconds": 0}

    monkeypatch.setattr("horen_opd.mquake_evaluation._generate_answer", generate)
    result = run_stream(
        RunConfig(dataset="mquake", n=100), engine, plan["edits"], {"train": []}, tmp_path, "id", plan=plan
    )
    assert result["n"] == 100 and result["n_training_facts"] == 1
    assert result["evaluation_checkpoints"] == [1]
    assert calls == [("Q1a", 64), ("Q1b", 64), ("Q2a", 64)]
    cases = json.loads((tmp_path / "multihop_cases.json").read_text())
    predictions = [
        json.loads(line) for line in (tmp_path / "multihop_predictions.jsonl").read_text().splitlines()
    ]
    assert summarize(cases, predictions)["official_split"] == result["metrics"]
    assert result["metrics"]["total"] == {"correct": 2, "count": 2, "accuracy": 1}
