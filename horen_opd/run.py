"""Train, checkpoint and evaluate one independent HoReN or HoReN+OPD run."""

import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import (
    atomic_json,
    canonical_hash,
    exclusive_run,
    load_checkpoint,
    now,
    restore_rng,
    save_checkpoint,
    sha256_file,
)
from .config import RunConfig
from .data import (
    canonical_sha256,
    dataset_input_paths,
    edit_evaluation_prompts,
    format_reasoning_prompt,
    history_replay,
    load_edits,
    load_reasoning,
    normalize_prompt,
)
from .evaluation import _edit_prediction, evaluate_edits, evaluate_unstructured_edits
from .mquake import load_plan
from .mquake_evaluation import evaluate as evaluate_mquake


def initialize_engine(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .training import OpdEngine

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, local_files_only=True, trust_remote_code=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=getattr(torch, config.dtype),
        device_map={"": config.device},
    )
    return OpdEngine(model, tokenizer, config.hparams, config.engine_config())


def _model_files(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir() or not list(root.glob("*.safetensors")):
        raise ValueError(f"Prepared local safetensors model required: {root}")
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".model"}
    }


def prepare_inputs(config):
    """Validate data and bind the run to content hashes before allocating the model."""
    config.validate()
    plan = load_plan(config.data_path) if config.dataset == "mquake" else None
    edits = plan["edits"] if plan else load_edits(config.data_path, config.n, config.dataset)
    if plan:
        data_hashes = {"plan": sha256_file(config.data_path)}
        if config.n != plan["setting"]:
            raise ValueError("MQuAKE n must equal the official setting in the plan")
    else:
        data_hashes = {
            name: sha256_file(path)
            for name, path in dataset_input_paths(config.data_path, config.dataset).items()
        }
    reasoning = {"train": []}
    if config.method == "preserve":
        if plan:
            reasoning = json.loads(Path(config.reasoning_path).read_text())
            if reasoning.get("audit", {}).get("train_sha256") != canonical_sha256(reasoning["train"]):
                raise ValueError("MQuAKE preservation pool hash mismatch")
            if reasoning["audit"].get("plan_hashes", {}).get(str(plan["setting"])) != canonical_sha256(plan):
                raise ValueError("Preservation pool was not prepared against this MQuAKE plan")
        else:
            reasoning = load_reasoning(config.reasoning_path)
            manifest = reasoning["manifest"]
            if (
                manifest.get("editing_dataset") != config.dataset
                or manifest.get("editing_inputs_sha256") != data_hashes
                or manifest.get("editing_n_excluded", 0) < config.n
            ):
                raise ValueError("Preservation pool does not cover this editing stream")
        if not reasoning["train"]:
            raise ValueError("Empty preservation pool")
        forbidden = {normalize_prompt(p) for p in edit_evaluation_prompts(edits)}
        if plan:
            forbidden.update(normalize_prompt(q) for c in plan["cases"] for q in c["questions"])
        if any(normalize_prompt(r["prompt"]) in forbidden for r in reasoning["train"]):
            raise ValueError("Preservation pool overlaps evaluation prompts")
        for row in reasoning["train"]:
            format_reasoning_prompt(row)
    model_files = _model_files(config.model_path)
    from .provenance import QWEN_REVISION, QWEN_WEIGHT_SHA256

    if config.model_revision == QWEN_REVISION:
        if any(model_files.get(name) != digest for name, digest in QWEN_WEIGHT_SHA256.items()):
            raise ValueError("Model weights differ from the pinned Qwen2.5-7B-Instruct revision")
    similarity = _model_files(config.similarity_model_path) if config.dataset == "unke" else None
    if similarity is not None:
        for name, version in (("rouge", "1.0.1"), ("sentence-transformers", "3.2.1")):
            if importlib.metadata.version(name) != version:
                raise ValueError(f"UnKE requires {name}=={version}")
    inputs = {
        "dataset": config.dataset,
        "data_sha256": data_hashes,
        "model_files": model_files,
        "similarity_files": similarity,
        "reasoning_sha256": sha256_file(config.reasoning_path) if config.method == "preserve" else None,
        "code_sha256": {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
    }
    return edits, reasoning, plan, inputs


def preservation_prompts(edits, index, reasoning, sampling):
    groups = defaultdict(list)
    for row in reasoning["train"]:
        groups[row["task"]].append(row)
    tasks = sorted(groups)
    history = history_replay(edits[:index], edits[index])
    prompts = [sampling.choice(history)["prompt"]] if history else []
    while len(prompts) < 2:
        task = tasks[(index + len(prompts)) % len(tasks)]
        prompts.append(format_reasoning_prompt(sampling.choice(groups[task])))
    return prompts


def metric_view(dataset, result):
    """Expose named Rel/Gen/Loc without inventing an UnKE locality score."""
    if dataset == "zsre":
        m = result["metrics"]
        return {"Rel": m["rewrite"], "Gen": m["rephrase"], "Loc": m["locality"]}
    if dataset == "unke":
        m = result["native_metrics"]
        return {"Rel": m["Original"], "Gen": m["Para"], "Loc": None, "Sub": m["Sub"]}
    return result["official_split"]


def run_stream(config, engine, edits, reasoning, directory, identity, plan=None, resume=False):
    """Resume only complete edit boundaries; pending evaluations are repeated safely."""
    directory = Path(directory)
    sampling = random.Random(config.seed + 2718)
    state = {
        "edit_index": 0,
        "phase": "ready",
        "edit_logs": [],
        "boundaries": [],
        "sampling_rng": sampling.getstate(),
        "training_seconds": 0.0,
        "evaluation_seconds": 0.0,
    }
    checkpoint = directory / "checkpoints"
    if resume:
        saved = load_checkpoint(checkpoint / "latest.json", identity)
        engine.restore(saved["engine"])
        state = saved["run"]
        sampling.setstate(state["sampling_rng"])
        restore_rng(saved["rng"])
    elif config.dataset == "zsre":
        started = time.monotonic()
        with engine.original_context():
            original = evaluate_edits(engine, edits)
        atomic_json(directory / "original.json", original)
        state["evaluation_seconds"] += time.monotonic() - started
    else:
        atomic_json(directory / "original.json", {"status": "not_evaluated"})
    original_locality = None
    if config.dataset == "zsre":
        original_locality = json.loads((directory / "original.json").read_text())["original_locality"]
    if not resume:
        save_checkpoint(checkpoint, engine.snapshot(), state, identity)
    if state["phase"] == "finished":
        return json.loads((directory / "final.json").read_text())
    points = set(config.evaluation_checkpoints) if plan is None else {len(edits)}

    def save():
        state["sampling_rng"] = sampling.getstate()
        save_checkpoint(checkpoint, engine.snapshot(), state, identity)

    def evaluate_boundary():
        n = state["edit_index"]
        started = time.monotonic()
        if plan:
            single = [
                {"case_id": e["case_id"], **_edit_prediction(engine, e["prompt"], e["target_new"])}
                for e in edits
            ]
            atomic_json(directory / "singlehop_predictions.json", single)
            pred_path = directory / "multihop_predictions.jsonl"
            if pred_path.exists():
                pred_path.unlink()
            result = evaluate_mquake(
                engine, plan["cases"], directory, identity, config.multihop_max_new_tokens
            )
            atomic_json(directory / "multihop_cases.json", plan["cases"])
            result["singlehop_rewrite_token_accuracy"] = sum(r["accuracy"] for r in single) / len(single)
        elif config.dataset == "unke":
            result = evaluate_unstructured_edits(
                engine, edits[:n], config.similarity_model_path, config.unstructured_max_new_tokens
            )
        else:
            result = evaluate_edits(engine, edits[:n], original_locality)
        atomic_json(directory / "evaluations" / f"{n:04d}.json", result)
        state["evaluation_seconds"] += time.monotonic() - started
        state["boundaries"].append(n)
        state["phase"] = "ready"
        save()
        print(
            json.dumps(
                {"event": "evaluation", "edit_index": n, "metrics": metric_view(config.dataset, result)}
            ),
            flush=True,
        )
        return result

    if (
        state["phase"] == "edited"
        and state["edit_index"] in points
        and state["edit_index"] not in state["boundaries"]
    ):
        evaluate_boundary()
    for index in range(state["edit_index"], len(edits)):
        prompts = (
            preservation_prompts(edits, index, reasoning, sampling) if config.method == "preserve" else []
        )
        started = time.monotonic()
        log = engine.edit(edits[index], prompts, preserve=config.method == "preserve")
        state["training_seconds"] += time.monotonic() - started
        state["edit_logs"].append({"case_id": edits[index]["case_id"], **log})
        state.update(edit_index=index + 1, phase="edited")
        if (index + 1) % config.checkpoint_every == 0 or index + 1 in points:
            save()
        atomic_json(
            directory / "progress.json",
            {"phase": "training", "edit_index": index + 1, "target_edits": len(edits), "updated_at": now()},
        )
        print(json.dumps({"event": "edit", "completed": index + 1, "total": len(edits)}), flush=True)
        if index + 1 in points:
            evaluate_boundary()
    result = json.loads((directory / "evaluations" / f"{len(edits):04d}.json").read_text())
    final = {
        "status": "completed",
        "identity": identity,
        "dataset": config.dataset,
        "method": config.method,
        "n": config.n,
        "n_training_facts": len(edits),
        "metrics": metric_view(config.dataset, result),
        "evaluation": result.get("metrics", result),
        "evaluation_checkpoints": state["boundaries"],
        "stats": engine.stats(),
        "training_seconds": state["training_seconds"],
        "evaluation_seconds": state["evaluation_seconds"],
        "finished_at": now(),
    }
    atomic_json(directory / "edit_logs.json", state["edit_logs"])
    atomic_json(directory / "final.json", final)
    state["phase"] = "finished"
    save()
    atomic_json(
        directory / "progress.json", {"phase": "finished", "edit_index": len(edits), "updated_at": now()}
    )
    return final


def artifacts(directory):
    return {
        str(p.relative_to(directory)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", ".lock"} and not p.name.startswith(".")
    }


def run(config, resume=False, check=False):
    edits, reasoning, plan, inputs = prepare_inputs(config)
    identity = canonical_hash({"config": config.to_dict(), "inputs": inputs})
    run_id = f"{config.dataset}_{config.method}_{'e' if plan else 'n'}{config.n}_s{config.seed}"
    directory = Path(config.output_dir) / run_id
    if check:
        return {
            "status": "checked",
            "run_id": run_id,
            "identity": identity,
            "training_facts": len(edits),
            "evaluation_cases": len(plan["cases"]) if plan else len(edits),
        }
    with exclusive_run(directory):
        manifest_path = directory / "manifest.json"
        if resume:
            old = json.loads(manifest_path.read_text())
            if old["identity"] != identity:
                raise ValueError("Cannot resume after changing config, model, data or code")
            if old["status"] == "completed":
                if old["artifacts"] != artifacts(directory):
                    raise ValueError("Completed run artifacts changed or are missing")
                return json.loads((directory / "final.json").read_text())
        elif any(p.name != ".lock" for p in directory.iterdir()):
            raise FileExistsError(f"Run already exists: {directory}; use --resume or a new output directory")
        manifest = {
            "status": "running",
            "identity": identity,
            "run_id": run_id,
            "config": config.to_dict(),
            "inputs": inputs,
            "started_at": old["started_at"] if resume else now(),
        }
        atomic_json(manifest_path, manifest)
        try:
            engine = initialize_engine(config)
            final = run_stream(config, engine, edits, reasoning, directory, identity, plan, resume)
            manifest.update(status="completed", finished_at=now(), artifacts=artifacts(directory))
            atomic_json(manifest_path, manifest)
            return final
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=now())
            atomic_json(manifest_path, manifest)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["baseline", "preserve"])
    parser.add_argument("--n", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate inputs without loading a model")
    args = parser.parse_args()
    config = RunConfig.load(
        args.config, method=args.method, n=args.n, device=args.device, output_dir=args.output_dir
    )
    print(json.dumps(run(config, args.resume, args.check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
