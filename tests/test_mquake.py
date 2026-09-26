import json

import pytest

from horen_opd.mquake import (
    answer_matches,
    build_protocol,
    filter_reasoning_pool,
    format_question,
    load_plan,
    summarize,
)


def row(cid, labels, triple=("S", "R", "NEW"), path=None):
    path = path or [("A", "REL", "S"), triple]
    return {
        "case_id": cid,
        "split": {"100": labels},
        "edit_triples": [list(triple)],
        "requested_rewrite": [
            {
                "prompt": "{} works at",
                "subject": "Someone",
                "relation_id": triple[1],
                "target_new_id": triple[2],
                "target_new_str": triple[2],
                "question": "Where does Someone work?",
            }
        ],
        "questions": [f"Q{cid}a", f"Q{cid}b", f"Q{cid}c"],
        "answer": "OLD",
        "answer_alias": ["old alias"],
        "new_answer": "NEW",
        "new_answer_alias": ["new alias"],
        "new_triples": [list(t) for t in path],
        "orig_triples": [["A", "REL", "S"], ["S", "R", "OLD"]],
        "single_hops": [{"question": "single question", "cloze": "single cloze"}],
        "new_single_hops": [],
    }


def test_shared_facts_and_test_never_adds_training():
    rows = [
        row(1, ["train_edited", "test_edited", "test_edited_unique"]),
        row(2, ["train_edited"]),
        row(3, ["test_edited"]),
    ]
    p = build_protocol(rows, 100)
    assert len(p["edits"]) == 1
    assert p["edits"][0]["source_case_ids"] == [1, 2]
    assert [r["group"] for r in p["cases"]] == ["train_edited", "train_edited", "test_edited"]
    assert p["audit"]["groups"] == {"train_edited": 2, "test_edited": 1}
    assert "questions" not in p["edits"][0]


def test_test_only_fact_fails_instead_of_leaking():
    with pytest.raises(ValueError, match="absent from training"):
        build_protocol([row(1, ["train_edited"]), row(2, ["test_edited"], ("X", "R", "NEW"))], 100)


def test_conflicting_training_facts_fail():
    with pytest.raises(ValueError, match="Conflicting training"):
        build_protocol([row(1, ["train_edited"]), row(2, ["train_edited"], ("S", "R", "OTHER"))], 100)


def test_path_conflict_kept_official_and_excluded_audited():
    bad = row(2, ["test_unedited"])
    p = build_protocol([row(1, ["train_edited"]), bad], 100)
    assert p["audit"]["path_conflicts"][0]["case_id"] == 2
    scores = summarize(p["cases"], [{"case_id": 1, "answers": ["NEW"]}, {"case_id": 2, "answers": ["OLD"]}])
    assert scores["official_split"]["total"]["count"] == 2
    assert scores["path_consistent_subset"]["total"]["count"] == 1
    assert scores["path_consistent_subset"]["unedited"]["accuracy"] is None


def test_any3_alias_case_exact_and_no_substring():
    p = build_protocol([row(1, ["train_edited"])], 100)
    case = p["cases"][0]
    assert answer_matches("NeW ALIAS", case)
    assert not answer_matches("The answer is NEW", case)
    assert not answer_matches("NEW YORK", case)
    s = summarize(p["cases"], [{"case_id": 1, "answers": ["bad", "new alias"]}])
    assert s["official_split"]["total"]["accuracy"] == 1
    with pytest.raises(ValueError, match="all three"):
        summarize(p["cases"], [{"case_id": 1, "answers": ["bad"]}])
    for preds in ([], [{"case_id": 1, "answers": ["NEW"]}] * 2):
        with pytest.raises(ValueError, match="case IDs"):
            summarize(p["cases"], preds)


def test_all_failed_attempts_score_zero():
    p = build_protocol([row(1, ["train_edited"])], 100)
    s = summarize(p["cases"], [{"case_id": 1, "answers": ["bad", "wrong", "no"]}])
    assert s["official_split"]["total"] == {"correct": 0, "count": 1, "accuracy": 0.0}


def test_no_answers_or_intermediate_gold_in_question_prompt():
    q = format_question("Who created this?")
    assert "Who created this?" in q
    assert "NEW" not in q
    assert q.endswith("<|im_start|>assistant\n")


def test_retention_pool_excludes_questions_and_clozes():
    rows = [row(1, ["train_edited"])]
    plan = build_protocol(rows, 100)
    r = {"train": [{"prompt": " q1a "}, {"prompt": "SINGLE CLOZE"}, {"prompt": "retained"}]}
    filtered = filter_reasoning_pool(r, rows, plan["edits"])
    assert filtered["train"] == [{"prompt": "retained"}]


def test_plan_content_identity(tmp_path):
    plan = build_protocol([row(1, ["train_edited"])], 100)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert load_plan(path) == plan
    plan["cases"][0]["answer"] = "tampered"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="modified"):
        load_plan(path)
