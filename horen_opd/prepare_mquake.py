"""Prepare CF6334 official splits and an isolated OPD preservation pool."""

import argparse
import json
from pathlib import Path

from .checkpoint import atomic_json, sha256_file
from .data import canonical_sha256, load_reasoning
from .mquake import build_protocol, filter_reasoning_pool


def prepare(input_path, reasoning_path, output_dir, settings):
    source = Path(input_path)
    if source.suffix == ".parquet":
        import pyarrow.parquet as parquet

        rows = parquet.read_table(source).to_pylist()
    else:
        rows = json.loads(source.read_text())
    plans = {s: build_protocol(rows, s) for s in settings}
    reasoning = load_reasoning(reasoning_path)
    pool = filter_reasoning_pool(reasoning, rows, [e for p in plans.values() for e in p["edits"]])
    pool["audit"]["plan_hashes"] = {str(s): canonical_sha256(p) for s, p in plans.items()}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for setting, plan in plans.items():
        atomic_json(output / f"plan_{setting}.json", plan)
    atomic_json(output / "reasoning_train.json", pool)
    atomic_json(
        output / "provenance.json",
        {
            "source_sha256": sha256_file(source),
            "reasoning_source_sha256": sha256_file(reasoning_path),
            "official_code_commit": "349dcc50460c251ee09a6804aab444508a016485",
            "settings": {str(s): p["audit"] for s, p in plans.items()},
            "reasoning_audit": pool["audit"],
        },
    )
    return {str(s): p["audit"] for s, p in plans.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--settings", nargs="+", type=int, choices=[100, 1000, 3000, 6334], default=[100, 1000]
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.reasoning, args.output_dir, args.settings), indent=2))


if __name__ == "__main__":
    main()
