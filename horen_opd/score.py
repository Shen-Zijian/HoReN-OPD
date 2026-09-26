"""Recalculate metrics from saved predictions without rerunning the edited LLM."""

import argparse
import json
from pathlib import Path
from statistics import fmean

from .checkpoint import atomic_json
from .evaluation import (
    _mean,
    _native_rouge,
    _rouge_recall,
    _safe_bleu_like,
    _sentence_embedding_similarity,
    token_agreement,
)
from .mquake import summarize


def score_zsre(evaluation):
    rows = []
    for row in evaluation["individuals"]:
        raw = row["raw"]
        scores = {}
        for name in ("rewrite", "rephrase"):
            values = []
            for item in raw.get(name, []):
                target, prediction = item["target_token_ids"], item["predicted_token_ids"]
                values.append(
                    fmean(a == b for a, b in zip(target, prediction))
                    if target and len(target) == len(prediction)
                    else (0.0 if target else 1.0)
                )
            scores[name] = _mean(values)
        values = [
            token_agreement(p["locality_token_ids"], p["original_token_ids"])
            for group in raw["locality"].values()
            for p in group
            if "original_token_ids" in p
        ]
        scores["locality"] = _mean(values)
        rows.append(scores)
    return {
        label: _mean([r[name] for r in rows if r[name] is not None])
        for label, name in (("Rel", "rewrite"), ("Gen", "rephrase"), ("Loc", "locality"))
    }


def score_unke(evaluation, similarity_model_path, similarity_scorer=None):
    rows, references, predictions = [], [], []
    rouge = _native_rouge()
    for row in evaluation["individuals"]:
        raw, result = row["raw"], {}
        for group in ("original", "para"):
            target, prediction = raw[group]["target"], raw[group]["prediction"]
            result[group] = {
                "BLEU SCORE": _safe_bleu_like(target, prediction),
                **{
                    name: score
                    for name, score in zip(
                        ("ROUGE-1", "ROUGE-2", "ROUGE-L"), _rouge_recall(target, prediction, rouge).values()
                    )
                },
            }
            references.append(target)
            predictions.append(prediction if " " in prediction else prediction + " ")
        sub = [_rouge_recall(p["target"], p["prediction"], rouge) for p in raw["sub"]]
        result["sub"] = {
            name: _mean([p[key] for p in sub])
            for name, key in (("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"))
        }
        rows.append(result)
    if rows:
        values = (
            similarity_scorer(references, predictions)
            if similarity_scorer
            else _sentence_embedding_similarity(references, predictions, similarity_model_path)
        )
        if len(values) != len(references):
            raise ValueError("Similarity score count mismatch")
        for index, value in enumerate(values):
            import math

            value = float(value)
            if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
                raise ValueError("Invalid semantic cosine")
            rows[index // 2]["original" if index % 2 == 0 else "para"]["Bert Score"] = min(
                1.0, max(-1.0, value)
            )
    result = {
        label: {
            name: _mean([r[group][name] for r in rows if r[group][name] is not None])
            for name in (
                ["BLEU SCORE", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Bert Score"]
                if group != "sub"
                else ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
            )
        }
        for label, group in (("Rel", "original"), ("Gen", "para"), ("Sub", "sub"))
    }
    return {**result, "Loc": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["zsre", "unke", "mquake"], required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--cases", help="multihop_cases.json for MQuAKE")
    parser.add_argument("--similarity-model", help="Local MiniLM directory for UnKE")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.dataset == "mquake":
        if not args.cases:
            parser.error("MQuAKE requires --cases")
        predictions = [
            json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()
        ]
        result = summarize(json.loads(Path(args.cases).read_text()), predictions)
    else:
        data = json.loads(Path(args.predictions).read_text())
        data = data.get("editing", data)
        if args.dataset == "unke" and not args.similarity_model:
            parser.error("UnKE requires --similarity-model")
        result = score_zsre(data) if args.dataset == "zsre" else score_unke(data, args.similarity_model)
    if args.output:
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
