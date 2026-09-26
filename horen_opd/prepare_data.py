"""Explicit online acquisition, followed by separate offline preparation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

from .data import (
    BBH_TASKS,
    EDIT_DATASETS,
    canonical_sha256,
    dataset_input_paths,
    edit_evaluation_prompts,
    load_edits,
    prepare_reasoning,
    save_reasoning,
)


GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"


def _commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("revision must be an immutable 40-character commit SHA")
    return value


def _get(url: str) -> bytes:
    import requests

    response = requests.get(url, headers={"User-Agent": "HoReN-OPD-reproducible-data/1"}, timeout=60)
    response.raise_for_status()
    return response.content


def download_sources(
    output_dir: Path, gsm_revision: str = GSM_REVISION, bbh_revision: str = BBH_REVISION
) -> Path:
    """Acquire pinned official files only. Requires pyarrow, never HF remote code."""
    _commit(gsm_revision)
    _commit(bbh_revision)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "Data acquisition requires pyarrow; install the project's data extras first"
        ) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "sources.json").exists():
        raise FileExistsError("sources.json already exists; use a new directory to preserve provenance")
    sources = {
        "gsm_train": [],
        "gsm_test": [],
        "bbh_tasks": {},
        "provenance": {
            "gsm_repository": "openai/gsm8k",
            "gsm_revision": gsm_revision,
            "gsm_config": "main",
            "bbh_repository": "suzgunmirac/BIG-Bench-Hard",
            "bbh_revision": bbh_revision,
            "files": [],
        },
    }

    def fetch(url: str, name: str) -> bytes:
        data = _get(url)
        destination = output_dir / name

        with destination.open("xb") as handle:
            handle.write(data)
        sources["provenance"]["files"].append(
            {
                "url": url,
                "filename": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
        return data

    for split in ("train", "test"):
        name = f"{split}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{gsm_revision}/main/{name}"
        raw = fetch(url, f"gsm8k_{name}")
        sources[f"gsm_{split}"] = parquet.read_table(io.BytesIO(raw)).to_pylist()
    for task in BBH_TASKS:
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{bbh_revision}/bbh/{task}.json"
        sources["bbh_tasks"][task] = json.loads(fetch(url, f"{task}.json"))["examples"]
    sources["provenance"]["parsed_sources_sha256"] = canonical_sha256(
        {k: sources[k] for k in ("gsm_train", "gsm_test", "bbh_tasks")}
    )
    target = output_dir / "sources.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return target


def build_reasoning_for_edits(
    source_json: Path, data_dir: Path, n: int = 1000, dataset: str = "zsre", seed: int = 42
) -> dict:
    """Build isolated retention data, hashing the entire selected source files."""
    dataset = dataset.lower()
    source_bytes = Path(source_json).read_bytes()
    sources = json.loads(source_bytes)
    parsed_hash = canonical_sha256({k: sources[k] for k in ("gsm_train", "gsm_test", "bbh_tasks")})
    if parsed_hash != sources.get("provenance", {}).get("parsed_sources_sha256"):
        raise ValueError("Source JSON does not match its recorded parsed-source hash")
    requests = load_edits(data_dir, n, dataset)
    data = prepare_reasoning(
        sources["gsm_train"],
        sources["gsm_test"],
        sources["bbh_tasks"],
        seed=seed,
        excluded_prompts=edit_evaluation_prompts(requests),
    )
    inputs = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in dataset_input_paths(data_dir, dataset).items()
    }
    data["manifest"].update(
        {
            "provenance": sources["provenance"],
            "source_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "editing_dataset": dataset,
            "editing_n_excluded": n,
            "editing_inputs_sha256": inputs,
            "editing_requests_sha256": canonical_sha256(requests),
            "editing_case_ids": [request["case_id"] for request in requests],
            "editing_source_ids": [request.get("source_id", request["case_id"]) for request in requests],
        }
    )
    if dataset == "zsre":
        data["manifest"]["zsre_n_excluded"] = n
        data["manifest"]["zsre_inputs_sha256"] = inputs
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Explicitly download immutable official input files")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--gsm-revision", type=_commit, default=GSM_REVISION)
    download.add_argument("--bbh-revision", type=_commit, default=BBH_REVISION)
    build = subparsers.add_parser("build", help="Offline deterministic splits; no network is used")
    build.add_argument("--source-json", type=Path, required=True)
    build.add_argument("--dataset", choices=EDIT_DATASETS, default="zsre")
    build.add_argument(
        "--data-dir", type=Path, required=True, help="Dataset directory or exact UnKE JSON path"
    )
    build.add_argument(
        "--n", type=int, default=1000, help="Number of edit rows excluded from retention data (default: 1000)"
    )
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download_sources(args.output_dir, args.gsm_revision, args.bbh_revision))
        return 0
    data = build_reasoning_for_edits(args.source_json, args.data_dir, args.n, args.dataset, args.seed)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace prepared data: {args.output}")
    save_reasoning(data, args.output)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "counts": data["manifest"]["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
