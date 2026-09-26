"""Offline, reproducible data preparation for HoReN-OPD."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


BBH_TASKS = (
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
)
DATA_SCHEMA_VERSION = 1
EDIT_DATASETS = ("zsre", "unke")


def normalize_prompt(prompt: str) -> str:
    """Conservative exact-match normalization, not semantic deduplication."""
    return " ".join(unicodedata.normalize("NFKC", str(prompt)).casefold().split())


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def dataset_input_paths(data_dir: str | Path, dataset: str = "zsre") -> dict[str, Path]:
    """Resolve only the explicitly supported dataset files, never a glob."""
    dataset = dataset.lower()
    if dataset not in EDIT_DATASETS:
        raise ValueError(f"Unsupported edit dataset: {dataset}")
    root = Path(data_dir)
    if dataset == "zsre":
        return {name: root / name for name in ("zsre_edit_data.json", "zsre_train_data.json")}
    filename = "final_data_v3.json"
    if root.suffix.lower() == ".json":
        if root.name != filename:
            raise ValueError(f"{dataset} requires the selected file {filename}, not {root.name}")
        return {filename: root}
    candidates = [root / filename, root / "UnKE" / filename]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Ambiguous {dataset} input directory; provide the exact JSON path")
    return {filename: existing[0] if existing else candidates[0]}


def _positive_n(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")


def _text(row: Mapping[str, Any], key: str, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} requires nonempty string {key}")
    return value


def _source_rows(path: Path, n: int, dataset: str) -> list[Mapping[str, Any]]:
    _positive_n(n)
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{dataset} source must contain a JSON array")
    if len(rows) < n:
        raise ValueError(f"Requested {n} {dataset} edits but found {len(rows)} rows")
    for index, row in enumerate(rows[:n]):
        if not isinstance(row, dict):
            raise ValueError(f"{dataset} row {index} must be an object")
    return rows[:n]


def load_zsre(data_dir: str | Path, n: int) -> list[dict]:
    """Match the existing structured runner: answers[0], not the `alt` field."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    root = Path(data_dir)
    edits = _read_json(root / "zsre_edit_data.json")
    locality = _read_json(root / "zsre_train_data.json")
    if not isinstance(edits, list) or not isinstance(locality, list):
        raise ValueError("ZsRE source files must contain JSON arrays")
    if min(len(edits), len(locality)) < n:
        raise ValueError(f"Requested {n} edits but found {len(edits)} edits/{len(locality)} locality rows")
    requests = []
    for case_id, (row, loc) in enumerate(zip(edits[:n], locality[:n])):
        for key in ("src", "subject", "rephrase", "answers"):
            if key not in row:
                raise ValueError(f"ZsRE row {case_id} lacks {key}")
        if (
            not isinstance(row["answers"], list)
            or not row["answers"]
            or not isinstance(row["answers"][0], str)
        ):
            raise ValueError(f"ZsRE row {case_id} has no string answer")
        if row["subject"] not in row["src"]:
            raise ValueError(f"ZsRE subject is absent from prompt at row {case_id}")
        requests.append(
            {
                "case_id": case_id,
                "prompt": row["src"],
                "subject": row["subject"],
                "target_new": row["answers"][0],
                "ground_truth": "<|endoftext|>",
                "rephrase_prompt": row["rephrase"],
                "locality": {"neighborhood": {"prompt": loc["loc"], "ground_truth": loc["loc_ans"]}},
                "portability": {},
                "loc_prompt": loc["loc"],
            }
        )
    return requests


def qwen_unstructured_prompt(question: str) -> str:
    """Exact Qwen template in native AKEW_both.get_qwen_without_answer."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def load_unke(data_dir: str | Path, n: int) -> list[dict]:
    """Load native Qwen UnKE unstructured samples without fabricated locality."""
    path = dataset_input_paths(data_dir, "unke")["final_data_v3.json"]
    requests, source_ids = [], set()
    for case_id, row in enumerate(_source_rows(path, n, "UnKE")):
        location = f"UnKE row {case_id}"
        source_id = row.get("id")
        if isinstance(source_id, bool) or not isinstance(source_id, (int, str)):
            raise ValueError(f"{location} requires an integer or string source id")
        if source_id in source_ids:
            raise ValueError(f"{location} duplicates source id {source_id!r}")
        source_ids.add(source_id)
        raw_prompt, raw_rephrase, raw_answer = (
            _text(row, key, location) for key in ("question", "para_question", "answer")
        )
        sub_questions, sub_answers = row.get("sub_question"), row.get("sub_answer")
        if not isinstance(sub_questions, list) or not isinstance(sub_answers, list):
            raise ValueError(f"{location} requires sub_question/sub_answer lists")
        if len(sub_questions) != len(sub_answers):
            raise ValueError(f"{location} has mismatched sub_question/sub_answer counts")
        for index, (question, answer) in enumerate(zip(sub_questions, sub_answers)):
            _text({"question": question, "answer": answer}, "question", f"{location} sub {index}")
            _text({"answer": answer}, "answer", f"{location} sub {index}")
        mmlu_questions = row.get("mmlu_questions", [])
        if not isinstance(mmlu_questions, list) or any(not isinstance(q, str) for q in mmlu_questions):
            raise ValueError(f"{location} mmlu_questions must be a list of strings")
        prompt, rephrase = (qwen_unstructured_prompt(q) for q in (raw_prompt, raw_rephrase))
        questions = [qwen_unstructured_prompt(q) for q in sub_questions]
        target = raw_answer + "<|im_end|>"
        requests.append(
            {
                "case_id": case_id,
                "source_id": source_id,
                "source_index": case_id,
                "dataset": "unke",
                "edit_format": "unstructured",
                "prompt": prompt,
                "target_new": target,
                "ground_truth": None,
                "rephrase_prompt": rephrase,
                "locality": {},
                "portability": {"sub": {"prompt": questions, "ground_truth": list(sub_answers)}}
                if questions
                else {},
                "id": case_id,
                "question": prompt,
                "para_question": rephrase,
                "answer": target,
                "sub_question": questions,
                "sub_answer": list(sub_answers),
                "raw_prompt": raw_prompt,
                "raw_rephrase_prompt": raw_rephrase,
                "raw_answer": raw_answer,
                "raw_evaluation_prompts": [raw_prompt, raw_rephrase, *sub_questions, *mmlu_questions],
            }
        )
    return requests


def load_edits(data_dir: str | Path, n: int, dataset: str = "zsre") -> list[dict]:
    """Dataset-aware dispatch; ZsRE keeps its original request representation."""
    loaders = {"zsre": load_zsre, "unke": load_unke}
    try:
        loader = loaders[dataset.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported edit dataset: {dataset}") from error
    return loader(data_dir, n)


def edit_evaluation_prompts(requests: Iterable[Mapping[str, Any]]) -> list[str]:
    """Prompts to exclude from reasoning retention, including held-out probes."""
    result: list[str] = []

    def append(value: Any) -> None:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                append(item)

    for request in requests:
        append(request["prompt"])
        append(request.get("rephrase_prompt"))
        append(request.get("raw_evaluation_prompts"))
        for group in ("locality", "portability"):
            for probe in request.get(group, {}).values():
                append(probe.get("prompt"))
    return result


def history_replay(history: Sequence[Mapping[str, Any]], current_request: Mapping[str, Any]) -> list[dict]:
    """Return past, non-conflicting original prompts; most recent target wins."""
    current_id = current_request.get("case_id")
    current_prompt = normalize_prompt(current_request["prompt"])
    latest: dict[str, dict] = {}
    ordered = sorted(enumerate(history), key=lambda item: item[1].get("case_id", item[0]))
    for _, request in ordered:
        if current_id is not None:
            if "case_id" not in request:
                raise ValueError("Replay requests must have case_id when the current request does")
            if request["case_id"] >= current_id:
                continue
        normalized = normalize_prompt(request["prompt"])
        if normalized == current_prompt:
            continue
        latest[normalized] = {
            key: copy.deepcopy(request[key])
            for key in ("case_id", "prompt", "target_new", "subject")
            if key in request
        }
    return list(latest.values())


def format_reasoning_prompt(record: Mapping[str, Any]) -> str:
    """Shared train/evaluation template; it never embeds the answer."""
    task = record["task"]
    if task == "gsm8k":
        instruction = (
            "Solve the problem step by step. End your answer with #### followed by the final number."
        )
    elif task in BBH_TASKS:
        instruction = (
            "Solve the logical deduction problem. End your answer with the option letter "
            "in parentheses, for example (A)."
        )
    else:
        raise ValueError(f"Unsupported reasoning task: {task}")
    return f"{instruction}\n\n{record['prompt']}\n\nAnswer:"


def _records(rows: Sequence[Mapping[str, Any]], task: str, source_split: str) -> list[dict]:
    result = []
    for index, row in enumerate(rows):
        prompt_key, answer_key = ("question", "answer") if task == "gsm8k" else ("input", "target")
        prompt, reference = row[prompt_key], row[answer_key]
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(reference, str):
            raise ValueError(f"Invalid {task}/{source_split} row {index}")
        if task == "gsm8k":
            if "####" not in reference:
                raise ValueError(f"GSM8K answer lacks final #### field at {source_split}/{index}")
            answer = reference.rsplit("####", 1)[1].strip().replace(",", "")
            if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", answer):
                raise ValueError(f"Invalid numeric GSM8K answer at {source_split}/{index}: {answer!r}")
        else:
            answer = reference.strip()
            if not re.fullmatch(r"\([A-Z]\)", answer):
                raise ValueError(f"Invalid BBH answer at {task}/{index}: {answer!r}")
        result.append(
            {
                "id": f"{task}/{source_split}/{index:06d}",
                "task": task,
                "prompt": prompt,
                "answer": answer,
                "reference_answer": reference,
                "source_split": source_split,
                "source_index": index,
                "prompt_sha256": hashlib.sha256(normalize_prompt(prompt).encode()).hexdigest(),
            }
        )
    return result


def prepare_reasoning(
    gsm_train: Sequence[Mapping[str, Any]],
    gsm_test: Sequence[Mapping[str, Any]],
    bbh_tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    seed: int = 42,
    gsm_train_size: int = 1024,
    gsm_dev_size: int = 256,
    excluded_prompts: Iterable[str] = (),
) -> dict:
    """Prepare disjoint, deterministic retention/dev/test data from offline rows."""
    if set(bbh_tasks) != set(BBH_TASKS):
        raise ValueError(f"Expected exactly these BBH tasks: {BBH_TASKS}")
    if min(gsm_train_size, gsm_dev_size) < 1:
        raise ValueError("GSM8K train/dev sizes must be positive")
    excludes = {normalize_prompt(p) for p in excluded_prompts}
    seen: set[str] = set()
    dropped: list[dict] = []
    result: dict[str, Any] = {"train": [], "dev": [], "test": []}

    official_test = _records(gsm_test, "gsm8k", "test")
    for record in official_test:
        key = normalize_prompt(record["prompt"])
        if key in seen:
            raise ValueError(
                "Official GSM8K test contains duplicate normalized prompts; refusing to alter it"
            )
        if key in excludes:
            raise ValueError(
                "Official GSM8K test overlaps an excluded edit/evaluation prompt; "
                "refusing a contaminated experiment or silently altering the official test"
            )
        seen.add(key)
    result["test"].extend(official_test)

    def unique(records: list[dict]) -> list[dict]:
        kept = []
        for record in records:
            key = normalize_prompt(record["prompt"])
            if key in seen or key in excludes:
                dropped.append({"id": record["id"], "reason": "duplicate_or_excluded_prompt"})
            else:
                seen.add(key)
                kept.append(record)
        return kept

    for task in BBH_TASKS:
        rows = unique(_records(bbh_tasks[task], task, "official"))
        if len(rows) < 5:
            raise ValueError(f"Too few unique examples in {task} for 40/20/40 splitting")
        random.Random(seed).shuffle(rows)
        n_train, n_dev = len(rows) * 2 // 5, len(rows) // 5
        result["train"].extend(rows[:n_train])
        result["dev"].extend(rows[n_train : n_train + n_dev])
        result["test"].extend(rows[n_train + n_dev :])

    rows = unique(_records(gsm_train, "gsm8k", "train"))
    if len(rows) < gsm_train_size + gsm_dev_size:
        raise ValueError(f"Need {gsm_train_size + gsm_dev_size} unique GSM8K train rows; have {len(rows)}")
    random.Random(seed).shuffle(rows)
    result["train"][0:0] = rows[:gsm_train_size]
    result["dev"][0:0] = rows[gsm_train_size : gsm_train_size + gsm_dev_size]
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "seed": seed,
        "gsm_train_size": gsm_train_size,
        "gsm_dev_size": gsm_dev_size,
        "bbh_split": "40/20/40_after_normalized_exact_deduplication",
        "split_rng": "Independent random.Random(seed) for each task",
        "deduplication": "Unicode NFKC + casefold + whitespace collapse; not semantic deduplication",
        "gsm_official_test_preserved": True,
        "excluded_prompts_sha256": canonical_sha256(sorted(excludes)),
        "excluded_prompt_count": len(excludes),
        "test_exclusion_overlap_count": sum(normalize_prompt(r["prompt"]) in excludes for r in official_test),
        "dropped": dropped,
        "unused_gsm_train_ids": [r["id"] for r in rows[gsm_train_size + gsm_dev_size :]],
        "inputs_sha256": {
            "gsm_train": canonical_sha256(gsm_train),
            "gsm_test": canonical_sha256(gsm_test),
            "bbh": {task: canonical_sha256(bbh_tasks[task]) for task in BBH_TASKS},
        },
        "split_ids": {split: [r["id"] for r in result[split]] for split in ("train", "dev", "test")},
        "counts": {
            split: {task: sum(r["task"] == task for r in result[split]) for task in ("gsm8k",) + BBH_TASKS}
            for split in ("train", "dev", "test")
        },
    }
    manifest["splits_sha256"] = canonical_sha256(result)
    result["manifest"] = manifest
    return result


def _validate_reasoning(payload: Mapping[str, Any]) -> None:
    if payload.get("manifest", {}).get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("Unsupported reasoning dataset schema")
    splits = {key: payload[key] for key in ("train", "dev", "test")}
    if canonical_sha256(splits) != payload["manifest"].get("splits_sha256"):
        raise ValueError("Reasoning split content hash mismatch")
    ids, prompts = set(), set()
    for split, records in splits.items():
        if not isinstance(records, list) or not records:
            raise ValueError(f"Reasoning {split} must be a nonempty list")
        if [r["id"] for r in records] != payload["manifest"]["split_ids"][split]:
            raise ValueError(f"Reasoning {split} IDs disagree with manifest")
        for record in records:
            prompt = normalize_prompt(record["prompt"])
            if record["id"] in ids or prompt in prompts:
                raise ValueError("Duplicate ID or normalized prompt across reasoning splits")
            ids.add(record["id"])
            prompts.add(prompt)
            format_reasoning_prompt(record)


def save_reasoning(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a self-validating dataset; no pickle or remote code."""
    _validate_reasoning(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_reasoning(path: str | Path) -> dict:
    """Load only prepared local JSON and verify split integrity/leakage guards."""
    payload = _read_json(path)
    _validate_reasoning(payload)
    return payload
