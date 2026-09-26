import json
import subprocess
import sys

import pytest

from horen_opd.data import (
    BBH_TASKS,
    edit_evaluation_prompts,
    format_reasoning_prompt,
    history_replay,
    load_reasoning,
    load_zsre,
    normalize_prompt,
    prepare_reasoning,
    save_reasoning,
)


def fixture_sources():
    train = [{"question": f"GSM train question {i}", "answer": f"Reasoning {i}. #### {i}"} for i in range(20)]
    test = [
        {"question": f"GSM official test question {i}", "answer": f"Reasoning. #### {i}"} for i in range(3)
    ]
    bbh = {task: [{"input": f"{task} problem {i}", "target": "(A)"} for i in range(10)] for task in BBH_TASKS}
    return train, test, bbh


def prepare_fixture(**kwargs):
    return prepare_reasoning(*fixture_sources(), gsm_train_size=6, gsm_dev_size=3, **kwargs)


def test_zsre_matches_existing_runner(tmp_path):
    edits = [
        {
            "src": f"Who is subject{i}?",
            "subject": f"subject{i}",
            "rephrase": f"Who was subject{i}?",
            "answers": [f"answer{i}"],
            "alt": "not_the_structured_runner_target",
        }
        for i in range(3)
    ]
    locality = [{"loc": f"Locality {i}?", "loc_ans": f"loc answer {i}"} for i in range(3)]
    (tmp_path / "zsre_edit_data.json").write_text(json.dumps(edits))
    (tmp_path / "zsre_train_data.json").write_text(json.dumps(locality))
    requests = load_zsre(tmp_path, 2)
    assert [r["case_id"] for r in requests] == [0, 1]
    assert requests[0]["target_new"] == "answer0"
    assert requests[0]["ground_truth"] == "<|endoftext|>"
    assert requests[1]["locality"]["neighborhood"]["prompt"] == "Locality 1?"
    assert requests[0]["portability"] == {}
    assert len(edit_evaluation_prompts(requests)) == 6
    with pytest.raises(ValueError, match="Requested"):
        load_zsre(tmp_path, 4)


def test_deterministic_splits_counts_and_official_test():
    a, b = prepare_fixture(), prepare_fixture()
    assert a == b
    assert a != prepare_fixture(seed=43)
    assert a["manifest"]["counts"]["train"]["gsm8k"] == 6
    assert a["manifest"]["counts"]["dev"]["gsm8k"] == 3
    assert a["manifest"]["counts"]["test"]["gsm8k"] == 3
    for task in BBH_TASKS:
        assert [a["manifest"]["counts"][s][task] for s in ("train", "dev", "test")] == [4, 2, 4]
    assert [r["source_index"] for r in a["test"] if r["task"] == "gsm8k"] == [0, 1, 2]
    all_ids = [r["id"] for split in ("train", "dev", "test") for r in a[split]]
    assert len(all_ids) == len(set(all_ids))


def test_dedupe_exclusion_and_no_cross_split_leakage():
    train, test, bbh = fixture_sources()
    train.insert(0, {"question": " GSM OFFICIAL test  question 0 ", "answer": "#### 12"})
    train.append(dict(train[4]))
    data = prepare_reasoning(
        train, test, bbh, gsm_train_size=6, gsm_dev_size=3, excluded_prompts=["gsm train question 1"]
    )
    prompts = [normalize_prompt(r["prompt"]) for s in ("train", "dev", "test") for r in data[s]]
    assert len(prompts) == len(set(prompts))
    assert "gsm train question 1" not in prompts
    assert len(data["manifest"]["dropped"]) == 3
    assert sum(r["task"] == "gsm8k" for r in data["test"]) == len(test)


def test_duplicate_official_test_is_error():
    train, test, bbh = fixture_sources()
    with pytest.raises(ValueError, match="Official GSM8K test"):
        prepare_reasoning(train, test + [test[0]], bbh, gsm_train_size=6, gsm_dev_size=3)


def test_official_test_overlap_with_edit_stream_is_error():
    train, test, bbh = fixture_sources()
    with pytest.raises(ValueError, match="overlaps an excluded"):
        prepare_reasoning(
            train,
            test,
            bbh,
            gsm_train_size=6,
            gsm_dev_size=3,
            excluded_prompts=[" GSM official test  question 0 "],
        )


def test_no_global_rng_mutation():
    import random

    random.seed(91)
    before = random.getstate()
    prepare_fixture()
    assert random.getstate() == before


def test_save_load_integrity(tmp_path):
    path = tmp_path / "data.json"
    data = prepare_fixture()
    save_reasoning(data, path)
    assert load_reasoning(path) == data
    content = json.loads(path.read_text())
    content["train"][0]["answer"] = "tampered"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_reasoning(path)


def test_history_latest_target_and_no_future_or_eval_fields():
    history = [
        {"case_id": 0, "prompt": "A question", "target_new": "old", "rephrase_prompt": "heldout"},
        {"case_id": 1, "prompt": "B question", "target_new": "b"},
        {"case_id": 2, "prompt": "a QUESTION", "target_new": "latest"},
        {"case_id": 4, "prompt": "Future", "target_new": "never"},
    ]
    result = history_replay(history, {"case_id": 3, "prompt": " b question ", "target_new": "new b"})
    assert result == [{"case_id": 2, "prompt": "a QUESTION", "target_new": "latest"}]
    result[0]["target_new"] = "modified"
    assert history[2]["target_new"] == "latest"


def test_shared_formatter_never_contains_reference():
    data = prepare_fixture()
    for record in data["train"]:
        formatted = format_reasoning_prompt(record)
        assert record["prompt"] in formatted
        if record["task"] == "gsm8k":
            assert record["reference_answer"] not in formatted
            assert "####" in formatted
        else:
            assert "(A)" in formatted
            assert "target" not in formatted


def test_prepare_cli_help_offline():
    result = subprocess.run(
        [sys.executable, "-m", "horen_opd.prepare_data", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "download" in result.stdout and "build" in result.stdout


def test_commit_rejects_mutable_reference():
    from horen_opd.prepare_data import _commit
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        _commit("main")
