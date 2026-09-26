"""Pinned MQuAKE-Remastered CF6334 preparation and case-level exact matching."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import fmean

from .data import canonical_sha256, normalize_prompt

PROTOCOL = "mquake-remastered-cf6334-official-split-qwen-zero-shot-direct-any3-alias-em-v1"
GROUPS = ("train_edited", "test_edited", "unedited")
QUESTION_INSTRUCTION = (
    "Answer the following question with only the answer entity. Do not include an explanation."
)


def format_question(question):
    return f"<|im_start|>user\n{QUESTION_INSTRUCTION}\n\n{question}<|im_end|>\n<|im_start|>assistant\n"


def build_protocol(rows, setting):
    """Implement the pinned HF official split, without importing its RAG pipeline."""
    if setting not in (100, 1000, 3000, 6334):
        raise ValueError("CF6334 has official settings 100, 1000, 3000, 6334")
    if len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source case ID")
    edits, index, relation_targets, train_ids = [], {}, {}, []
    for row in rows:
        if "train_edited" not in row["split"][str(setting)]:
            continue
        train_ids.append(row["case_id"])
        triples, requests = row["edit_triples"], row["requested_rewrite"]
        if len(triples) != len(requests):
            raise ValueError("Edit triple/request alignment mismatch")
        for triple, request in zip(triples, requests):
            triple = tuple(triple)
            if (triple[1], triple[2]) != (request["relation_id"], request["target_new_id"]):
                raise ValueError("Edit IDs disagree with request")
            sr = triple[:2]
            if sr in relation_targets and relation_targets[sr] != triple[2]:
                raise ValueError(f"Conflicting training facts: {sr}")
            relation_targets[sr] = triple[2]
            if triple in index:
                edits[index[triple]]["source_case_ids"].append(row["case_id"])
                continue
            prompt = request["prompt"].format(request["subject"])
            if not prompt.strip() or not request["target_new_str"].strip():
                raise ValueError("Empty edit prompt/target")
            index[triple] = len(edits)
            edits.append(
                {
                    "case_id": len(edits),
                    "prompt": prompt,
                    "subject": request["subject"],
                    "target_new": request["target_new_str"],
                    "triple": list(triple),
                    "source_case_ids": [row["case_id"]],
                }
            )
    if not edits:
        raise ValueError("No training facts")
    cases, anomalies = [], []
    for row in rows:
        labels = row["split"][str(setting)]

        group = (
            "train_edited"
            if "train_edited" in labels
            else "test_edited"
            if "test_edited" in labels
            else "unedited"
            if "test_unedited" in labels
            else None
        )
        if group is None:
            continue
        if group != "unedited" and "test_unedited" in labels:
            raise ValueError("Contradictory split labels")
        if len(row["questions"]) != 3 or any(
            not isinstance(q, str) or not q.strip() for q in row["questions"]
        ):
            raise ValueError("Each case must supply three nonempty questions")
        is_edited = group != "unedited"
        prefix = "new_" if is_edited else ""
        path = row["new_triples"] if is_edited else row["orig_triples"]
        conflicts = [
            list(t)
            for t in path
            if tuple(t[:2]) in relation_targets and relation_targets[tuple(t[:2])] != t[2]
        ]
        missing = [
            list(t) for t in row["edit_triples"] if is_edited and relation_targets.get(tuple(t[:2])) != t[2]
        ]

        if missing:
            raise ValueError(f"Case {row['case_id']} needs facts absent from training: {missing}")
        if conflicts:
            anomalies.append(
                {
                    "case_id": row["case_id"],
                    "group": group,
                    "conflicting_path_triples": conflicts,
                    "bank_targets": [relation_targets[tuple(t[:2])] for t in conflicts],
                }
            )
        cases.append(
            {
                "case_id": row["case_id"],
                "group": group,
                "edited": is_edited,
                "hops": len(path),
                "questions": row["questions"],
                "answer": row[prefix + "answer"],
                "answer_alias": row[prefix + "answer_alias"],
                "path_consistent": not conflicts,
            }
        )
    counts = dict(Counter(c["group"] for c in cases))
    return {
        "protocol": PROTOCOL,
        "setting": setting,
        "train_case_ids": train_ids,
        "edits": edits,
        "cases": cases,
        "audit": {
            "source_rows": len(rows),
            "train_cases": len(train_ids),
            "unique_edit_facts": len(edits),
            "evaluation_cases": len(cases),
            "groups": counts,
            "question_count_upper_bound": 3 * len(cases),
            "path_conflicts": anomalies,
            "excluded_by_official_split": len(rows) - len(cases),
            "training_facts_sha256": canonical_sha256(edits),
            "evaluation_cases_sha256": canonical_sha256(cases),
        },
    }


def filter_reasoning_pool(reasoning, rows, edits):
    forbidden = set()
    for row in rows:
        forbidden.update(normalize_prompt(q) for q in row["questions"])
        for key in ("single_hops", "new_single_hops"):
            for hop in row[key]:
                forbidden.add(normalize_prompt(hop["question"]))
                forbidden.add(normalize_prompt(hop["cloze"]))
        for request in row["requested_rewrite"]:
            forbidden.add(normalize_prompt(request["question"]))
    forbidden.update(normalize_prompt(e["prompt"]) for e in edits)
    train = [r for r in reasoning["train"] if normalize_prompt(r["prompt"]) not in forbidden]
    if not train:
        raise ValueError("Empty preservation training pool")
    return {
        "train": train,
        "audit": {
            "source_train_count": len(reasoning["train"]),
            "kept": len(train),
            "removed_exact_prompt_overlap": len(reasoning["train"]) - len(train),
            "exclusion": "NFKC casefold whitespace exact match against all MQuAKE questions and clozes; not semantic decontamination",
            "train_sha256": canonical_sha256(train),
        },
    }


def answer_matches(answer, case):
    return isinstance(answer, str) and answer.upper() in {
        a.upper() for a in [case["answer"], *case["answer_alias"]]
    }


def summarize(cases, predictions):
    expected = {c["case_id"] for c in cases}
    actual = [r["case_id"] for r in predictions]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Predictions have missing/duplicate/unexpected case IDs")
    by_id = {r["case_id"]: r for r in predictions}
    scores = []
    for c in cases:
        p = by_id[c["case_id"]]
        answers = p["answers"]
        if not 1 <= len(answers) <= 3:
            raise ValueError("Expected one to three attempted rephrases")
        correct = any(answer_matches(a, c) for a in answers)
        if not correct and len(answers) != 3:
            raise ValueError("A failed case must attempt all three questions")
        scores.append({**c, "correct": correct})

    def metric(items):
        return {
            "correct": sum(r["correct"] for r in items),
            "count": len(items),
            "accuracy": fmean(r["correct"] for r in items) if items else None,
        }

    def grouped(items):
        return {
            "total": metric(items),
            **{g: metric([r for r in items if r["group"] == g]) for g in GROUPS},
            "by_hop": {str(h): metric([r for r in items if r["hops"] == h]) for h in (2, 3, 4)},
        }

    return {
        "official_split": grouped(scores),
        "path_consistent_subset": grouped([r for r in scores if r["path_consistent"]]),
        "known_inconsistent_case_ids": [r["case_id"] for r in scores if not r["path_consistent"]],
        "scoring": "case-insensitive exact answer/alias match; any of 3 rephrases; stop after first success",
        "generation_calls": sum(len(r["answers"]) for r in predictions),
    }


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("Unknown MQuAKE protocol")
    for key in ("edits", "cases"):
        expected = plan["audit"]["training_facts_sha256" if key == "edits" else "evaluation_cases_sha256"]
        if canonical_sha256(plan[key]) != expected:
            raise ValueError("Prepared MQuAKE plan modified")
    return plan
