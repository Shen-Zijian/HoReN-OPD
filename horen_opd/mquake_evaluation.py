"""MQuAKE direct-answer inference and official case-level scoring."""

import json
import time
from .checkpoint import atomic_json, now
from .mquake import answer_matches, format_question, summarize


def _generate_answer(engine, question, max_new_tokens):
    from transformers import StoppingCriteria, StoppingCriteriaList

    prompt = format_question(question)
    prompt_len = len(engine.tokenizer.encode(prompt))

    class StopAtNewline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = engine.tokenizer.decode(input_ids[0, prompt_len:].tolist(), skip_special_tokens=True)
            return "\n" in text or "\r" in text

    started = time.monotonic()
    ids, length = engine._generate_ids(
        prompt,
        max_new_tokens,
        False,
        generation_overrides={
            "repetition_penalty": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
            "num_beams": 1,
            "stopping_criteria": StoppingCriteriaList([StopAtNewline()]),
        },
    )
    raw = engine.tokenizer.decode(ids[0, length:].tolist(), skip_special_tokens=True)

    answer = raw.split("\n", 1)[0].split("\r", 1)[0].strip()
    return answer, {"raw": raw, "tokens": len(ids[0]) - length, "elapsed_seconds": time.monotonic() - started}


def evaluate(engine, cases, directory, identity, max_new_tokens):
    predictions = []
    started = time.monotonic()
    with (directory / "multihop_predictions.jsonl").open("x") as f:
        for index, case in enumerate(cases):
            answers, attempts = [], []
            for qid, question in enumerate(case["questions"]):
                answer, detail = _generate_answer(engine, question, max_new_tokens)
                answers.append(answer)
                attempts.append({"qid": qid, "question": question, **detail})
                if answer_matches(answer, case):
                    break
            record = {
                "case_id": case["case_id"],
                "group": case["group"],
                "hops": case["hops"],
                "path_consistent": case["path_consistent"],
                "answers": answers,
                "attempts": attempts,
                "correct": any(answer_matches(a, case) for a in answers),
                "target": case["answer"],
                "aliases": case["answer_alias"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            predictions.append(record)
            atomic_json(
                directory / "progress.json",
                {
                    "phase": "multihop_evaluation",
                    "identity": identity,
                    "completed_cases": index + 1,
                    "target_cases": len(cases),
                    "updated_at": now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "generation_calls": sum(len(p["answers"]) for p in predictions),
                },
            )
            if (index + 1) % 25 == 0 or index + 1 == len(cases):
                print(
                    json.dumps({"event": "multihop", "completed": index + 1, "total": len(cases)}), flush=True
                )
    return summarize(cases, predictions)
