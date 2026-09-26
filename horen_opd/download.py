"""Explicit downloads of pinned models and CF6334 data."""

import argparse
import shutil
from pathlib import Path

from .checkpoint import atomic_json, sha256_file
from .provenance import MINILM_REVISION, MQUAKE_REVISION, QWEN_REVISION


def main():
    from huggingface_hub import hf_hub_download, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["qwen", "minilm", "mquake"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError("Use a new output directory; existing assets are never overwritten")
    if args.asset == "mquake":
        repo, revision = "henryzhongsc/MQuAKE-Remastered", MQUAKE_REVISION
        source = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename="data/CF6334-00000-of-00001.parquet",
        )
        destination.mkdir(parents=True)
        shutil.copyfile(source, destination / "CF6334.parquet")
    else:
        repo, revision = (
            ("Qwen/Qwen2.5-7B-Instruct", QWEN_REVISION)
            if args.asset == "qwen"
            else ("sentence-transformers/all-MiniLM-L6-v2", MINILM_REVISION)
        )
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=destination,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    files = {
        str(p.relative_to(destination)): sha256_file(p)
        for p in sorted(destination.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }
    atomic_json(
        destination / "download_manifest.json", {"repo_id": repo, "revision": revision, "files": files}
    )
    print(destination.resolve())


if __name__ == "__main__":
    main()
