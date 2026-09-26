"""Local, reproducible evaluation; no model judges or network calls."""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence


UNSTRUCTURED_KEYS = tuple(
    f"{group}_{metric}"
    for group, metrics in (
        ("original", ("bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity")),
        ("para", ("bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity")),
        ("sub", ("rouge1", "rouge2", "rougeL")),
    )
    for metric in metrics
)
UNSTRUCTURED_SIGNED_KEYS = ("original_semantic_similarity", "para_semantic_similarity")


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _pairs(prompts: Any, targets: Any) -> list[tuple[str, str]]:
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(targets, str):
        targets = [targets] * len(prompts)
    if not isinstance(prompts, (list, tuple)) or not isinstance(targets, (list, tuple)):
        raise ValueError("Prompts and targets must be strings or lists of strings")
    if len(prompts) != len(targets):
        raise ValueError("Prompt/target count mismatch; refusing silent zip truncation")
    if any(not isinstance(p, str) or not isinstance(t, str) for p, t in zip(prompts, targets)):
        raise ValueError("Every prompt and target must be a string")
    return list(zip(prompts, targets))


def _edit_prediction(engine: Any, prompt: str, target: str) -> dict[str, Any]:
    tokenizer = engine.tokenizer
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    budget = len(tokenizer.encode(" " + target, add_special_tokens=False))
    if budget < 1:
        raise ValueError("A target must allow at least one generation token")
    full_ids = list(engine.generate_tokens(prompt, max_new_tokens=budget, sample=False))
    prompt_length = len(tokenizer.encode(prompt))
    if len(full_ids) < prompt_length:
        raise ValueError("generate_tokens must return the full, untruncated prompt plus continuation")
    continuation_ids = full_ids[prompt_length:]
    prediction = tokenizer.decode(continuation_ids, skip_special_tokens=True).lstrip()
    predicted_ids = list(tokenizer.encode(prediction, add_special_tokens=False))[: len(target_ids)]

    accuracy = (
        fmean(float(a == b) for a, b in zip(target_ids, predicted_ids))
        if target_ids and len(target_ids) == len(predicted_ids)
        else (1.0 if not target_ids else 0.0)
    )

    locality_ids = full_ids[-len(target_ids) :]
    return {
        "prompt": prompt,
        "target": target,
        "prediction": prediction,
        "target_token_ids": target_ids,
        "predicted_token_ids": predicted_ids,
        "continuation_token_ids": continuation_ids,
        "locality_token_ids": locality_ids,
        "accuracy": accuracy,
    }


def token_agreement(candidate: Sequence[int], reference: Sequence[int]) -> float:
    """Upstream locality: aligned matches / maximum sequence length."""
    denominator = max(len(candidate), len(reference))
    return sum(a == b for a, b in zip(candidate, reference)) / denominator if denominator else 1.0


def evaluate_edits(
    engine: Any,
    requests: Sequence[Mapping[str, Any]],
    original_locality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score all supplied historical edits without ever adding codebook slots."""
    individuals: list[dict[str, Any]] = []
    locality_outputs: dict[str, Any] = {}

    port_names = sorted({name for request in requests for name in (request.get("portability") or {})})
    all_scores: dict[str, list[float]] = {
        k: []
        for k in (
            "rewrite",
            "rephrase",
            "locality",
            "portability",
            *(f"portability_{name}" for name in port_names),
        )
    }
    seen: set[str] = set()
    for request in requests:
        if "case_id" not in request:
            raise ValueError("Every edit request must have a stable case_id")
        case_id = request["case_id"]
        key = str(case_id)
        if key in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(key)
        raw: dict[str, Any] = {}
        scores: dict[str, list[float]] = {k: [] for k in all_scores}
        for metric, prompt_key in (("rewrite", "prompt"), ("rephrase", "rephrase_prompt")):
            prompts = request.get(prompt_key)
            if prompts is None:
                if metric == "rewrite":
                    raise ValueError(f"Missing rewrite prompt: {case_id}")
                continue
            predictions = [_edit_prediction(engine, p, t) for p, t in _pairs(prompts, request["target_new"])]
            raw[metric] = predictions
            scores[metric] = [item["accuracy"] for item in predictions]
        locality_outputs[key] = {}
        raw["locality"] = {}
        for name, group in (request.get("locality") or {}).items():
            predictions = [
                _edit_prediction(engine, p, t) for p, t in _pairs(group["prompt"], group["ground_truth"])
            ]
            tokens = [item["locality_token_ids"] for item in predictions]
            locality_outputs[key][name] = tokens
            if original_locality is not None:
                try:
                    reference = original_locality[key][name]
                except KeyError as exc:
                    raise ValueError(f"Missing original locality tokens for case {case_id}, {name}") from exc
                if len(reference) != len(tokens):
                    raise ValueError(f"Locality reference count mismatch for case {case_id}, {name}")
                for item, candidate_ids, reference_ids in zip(predictions, tokens, reference):
                    item["original_token_ids"] = list(reference_ids)
                    item["agreement"] = token_agreement(candidate_ids, reference_ids)
                    scores["locality"].append(item["agreement"])
            raw["locality"][name] = predictions
        raw["portability"] = {}
        for name, group in (request.get("portability") or {}).items():
            predictions = [
                _edit_prediction(engine, p, t) for p, t in _pairs(group["prompt"], group["ground_truth"])
            ]
            scores["portability"].extend(item["accuracy"] for item in predictions)
            scores[f"portability_{name}"].extend(item["accuracy"] for item in predictions)
            raw["portability"][name] = predictions
        per_case = {metric: _mean(values) for metric, values in scores.items()}

        for metric, value in per_case.items():
            if value is not None:
                all_scores[metric].append(value)
        individuals.append(
            {
                "case_id": case_id,
                "requested_rewrite": dict(request),
                "metrics": per_case,
                "raw": raw,
                "predictions": raw,
            }
        )
    return {
        "metrics": {metric: _mean(values) for metric, values in all_scores.items()},
        "individuals": individuals,
        "original_locality": locality_outputs,
        "metric_definition": "horen_autoregressive_token_accuracy_and_original_output_locality",
        "locality_status": "reference_capture_only"
        if original_locality is None
        else "original_output_agreement",
        "portability_status": "available" if all_scores["portability"] else "not_applicable",
    }


def _safe_bleu_like(reference_text: str, hypothesis_text: str) -> float:
    """Released UnKE evaluator's whitespace BLEU-1-like precision and penalty."""
    reference, hypothesis = reference_text.split(), hypothesis_text.split()
    if not reference or not hypothesis:
        return 0.0
    ref_counts, hyp_counts = Counter(reference), Counter(hypothesis)
    overlap = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
    penalty = 1.0 if len(hypothesis) > len(reference) else math.exp(1 - len(reference) / len(hypothesis))
    return float(penalty * overlap / len(hypothesis))


@lru_cache(maxsize=1)
def _native_rouge() -> Any:
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise RuntimeError("UnKE evaluation requires the pinned rouge==1.0.1 package") from exc
    return Rouge()


def _rouge_recall(reference: str, prediction: str, scorer: Any) -> dict[str, float]:
    if not prediction.strip() or not any(sentence.strip() for sentence in prediction.split(".")):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scored_prediction = prediction if " " in prediction else prediction + " "
    result = scorer.get_scores(scored_prediction, reference)[0]
    return {
        key: float(result[source]["r"])
        for key, source in (
            ("rouge1", "rouge-1"),
            ("rouge2", "rouge-2"),
            ("rougeL", "rouge-l"),
        )
    }


@lru_cache(maxsize=2)
def _local_sentence_model(model_path: str, device: str) -> Any:
    """Load only a provisioned local model, never resolve/download a Hub ID."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local UnKE similarity model directory is missing: {path}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("UnKE evaluation requires the pinned sentence-transformers package") from exc
    model = SentenceTransformer(str(path), device=device, local_files_only=True)
    model.eval()
    return model


def _sentence_embedding_similarity(
    references: Sequence[str], predictions: Sequence[str], model_path: str, device: str = "cpu"
) -> list[float]:
    """Paired MiniLM cosine, as upstream's diagonal cos_sim, without an N² matrix."""
    import torch
    from torch.nn import functional as F
    from .checkpoint import capture_rng, restore_rng

    state = capture_rng()
    try:
        model = _local_sentence_model(str(Path(model_path).expanduser().resolve()), device)
        with torch.inference_mode():
            reference_embeddings = model.encode(
                list(references), convert_to_tensor=True, show_progress_bar=False
            )
            prediction_embeddings = model.encode(
                list(predictions), convert_to_tensor=True, show_progress_bar=False
            )

            reference_embeddings = F.normalize(reference_embeddings, p=2, dim=1)
            prediction_embeddings = F.normalize(prediction_embeddings, p=2, dim=1)
            values = (reference_embeddings * prediction_embeddings).sum(dim=1)
        return values.detach().cpu().tolist()
    finally:
        restore_rng(state)


def _unstructured_prediction(engine: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    full_ids = list(engine.generate_tokens(prompt, max_new_tokens=max_new_tokens, sample=False))
    prompt_length = len(engine.tokenizer.encode(prompt))
    if len(full_ids) < prompt_length:
        raise ValueError("generate_tokens must return the full prompt plus continuation")
    continuation = full_ids[prompt_length:]
    return {
        "prompt": prompt,
        "prediction": engine.tokenizer.decode(continuation, skip_special_tokens=True),
        "continuation_token_ids": continuation,
        "max_new_tokens": max_new_tokens,
    }


def evaluate_unstructured_edits(
    engine: Any,
    requests: Sequence[Mapping[str, Any]],
    similarity_model_path: str,
    max_new_tokens: int = 512,
    *,
    similarity_scorer: Callable[[Sequence[str], Sequence[str]], Sequence[float]] | None = None,
    similarity_device: str = "cpu",
    rouge_scorer: Any = None,
) -> dict[str, Any]:
    """Native-aligned UnKE Rel/Gen/Port, with explicit unsupported locality."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    seen: set[str] = set()
    prepared: list[tuple[Mapping[str, Any], str, str, str, list[tuple[str, str]]]] = []
    for request in requests:
        if "case_id" not in request:
            raise ValueError("Every edit request must have a stable case_id")
        key = str(request["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate case_id: {request['case_id']}")
        seen.add(key)
        prompt = request.get("prompt", request.get("question"))
        para_prompt = request.get("rephrase_prompt", request.get("para_question"))
        target = request.get("target_new", request.get("answer"))
        if not all(isinstance(value, str) and value.strip() for value in (prompt, para_prompt, target)):
            raise ValueError(f"UnKE requires nonempty original/para prompts and answer for case {key}")
        for suffix in ("<|eot_id|>", "<|im_end|>"):
            if target.endswith(suffix):
                target = target[: -len(suffix)]
                break
        if not target.strip():
            raise ValueError(f"UnKE reference answer is empty after suffix removal: {key}")
        if request.get("locality"):
            raise ValueError("Native UnKE evaluation has no locality metric; do not pass locality probes")
        groups = request.get("portability") or {}
        if set(groups) - {"sub"}:
            raise ValueError("UnKE portability only supports the native sub-question group")
        sub = groups.get("sub")
        pairs = (
            _pairs(sub["prompt"], sub["ground_truth"])
            if sub is not None
            else _pairs(request.get("sub_question", []), request.get("sub_answer", []))
        )
        if any(not p.strip() or not t.strip() for p, t in pairs):
            raise ValueError(f"Empty UnKE sub-question or answer: {key}")
        prepared.append((request, prompt, para_prompt, target, pairs))

    scorer = rouge_scorer if rouge_scorer is not None else _native_rouge()
    individuals: list[dict[str, Any]] = []
    references: list[str] = []
    similarity_predictions: list[str] = []
    for request, prompt, para_prompt, target, sub_pairs in prepared:
        scores: dict[str, float | None] = {key: None for key in UNSTRUCTURED_KEYS}
        scores["locality"] = None
        raw: dict[str, Any] = {}
        for name, question in (("original", prompt), ("para", para_prompt)):
            prediction = _unstructured_prediction(engine, question, max_new_tokens)
            prediction["target"] = target
            raw[name] = prediction
            scores[f"{name}_bleu"] = _safe_bleu_like(target, prediction["prediction"])
            for metric, value in _rouge_recall(target, prediction["prediction"], scorer).items():
                scores[f"{name}_{metric}"] = value
            references.append(target)

            text = prediction["prediction"]
            similarity_predictions.append(text if " " in text else text + " ")
        raw["sub"] = []
        sub_scores: dict[str, list[float]] = {metric: [] for metric in ("rouge1", "rouge2", "rougeL")}
        for question, answer in sub_pairs:
            prediction = _unstructured_prediction(engine, question, max_new_tokens)
            prediction["target"] = answer
            prediction["metrics"] = _rouge_recall(answer, prediction["prediction"], scorer)
            raw["sub"].append(prediction)
            for metric, value in prediction["metrics"].items():
                sub_scores[metric].append(value)
        scores.update({f"sub_{metric}": _mean(values) for metric, values in sub_scores.items()})
        individuals.append(
            {
                "case_id": request["case_id"],
                "requested_rewrite": dict(request),
                "metrics": scores,
                "raw": raw,
                "predictions": raw,
            }
        )

    if references:
        values = list(
            similarity_scorer(references, similarity_predictions)
            if similarity_scorer is not None
            else _sentence_embedding_similarity(
                references, similarity_predictions, similarity_model_path, similarity_device
            )
        )
        if len(values) != len(references):
            raise ValueError("Semantic similarity result count mismatch")
        for index, value in enumerate(values):
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Invalid semantic similarity score") from exc
            if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
                raise ValueError("Semantic cosine must be finite and in [-1, 1]")

            value = min(1.0, max(-1.0, value))
            group = "original" if index % 2 == 0 else "para"
            individuals[index // 2]["metrics"][f"{group}_semantic_similarity"] = value

    metrics = {
        key: _mean([item["metrics"][key] for item in individuals if item["metrics"][key] is not None])
        for key in (*UNSTRUCTURED_KEYS, "locality")
    }
    labels = {
        "bleu": "BLEU SCORE",
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougeL": "ROUGE-L",
        "semantic_similarity": "Bert Score",
    }
    native_metrics = {
        group.capitalize(): {
            label: metrics[f"{group}_{metric}"]
            for metric, label in labels.items()
            if f"{group}_{metric}" in metrics
        }
        for group in ("original", "para", "sub")
    }
    return {
        "metrics": metrics,
        "native_metrics": native_metrics,
        "individuals": individuals,
        "original_locality": {},
        "locality_status": "not_applicable",
        "portability_status": "available"
        if any(item["raw"]["sub"] for item in individuals)
        else "not_applicable",
        "metric_definition": "horen_unke_whitespace_bleu1_like_rouge_1.0.1_recall_minilm_cosine;empty_prediction_overlap_zero",
        "semantic_similarity_definition": {
            "metric": "sentence_embedding_cosine",
            "native_label": "Bert Score",
            "is_bertscore": False,
            "model": str(similarity_model_path),
            "device": similarity_device,
            "float32_roundoff_clamp_tolerance": 0.000001,
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "batch_size": 1,
            "use_cache": False,
            "prompt_template": "native_qwen_unstructured_preformatted",
        },
    }
