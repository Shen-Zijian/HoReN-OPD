"""Portable configuration for independent sequential editing runs."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


def baseline_hparams():
    return {
        "inner_params": ["model.layers[24].mlp.down_proj.weight"],
        "adapter_mode": "value",
        "replacement": "replace_last",
        "n_iter": 50,
        "edit_lr": 1.0,
        "val_init": "cold",
        "normalize_codebook_keys": True,
        "query_selection_strategy": "last_60_perc_prompt_tokens_avg",
        "hopfield_retrieval_alpha": 0.1,
        "hopfield_retrieval_beta": 20.0,
        "hopfield_retrieval_eps": 1e-5,
        "hopfield_retrieval_max_iter": 1,
        "hopfield_key_match_threshold": 0.55,
        "batch_size": 1,
        "max_length": 512,
    }


@dataclass
class RunConfig:
    model_path: str = "models/qwen2.5-7b-instruct"
    model_revision: str = "a09a35458c702b33eeacc393d103063234e8bc28"
    dataset: str = "zsre"
    data_path: str = "data/ZsRE"
    reasoning_path: str | None = "data/reasoning_zsre.json"
    similarity_model_path: str | None = None
    output_dir: str = "runs"
    method: str = "baseline"
    n: int = 500
    seed: int = 42
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    eval_every: int = 100
    checkpoint_every: int = 1
    opd_steps: int = 1
    opd_lr: float = 1e-3
    kl_weight: float = 1.0
    rollout_max_new_tokens: int = 256
    kl_chunk_size: int = 32
    unstructured_max_new_tokens: int = 512
    multihop_max_new_tokens: int = 64
    hparams: dict = field(default_factory=baseline_hparams)

    def validate(self):
        if self.dataset not in {"zsre", "unke", "mquake"}:
            raise ValueError("dataset must be zsre, unke or mquake")
        if self.method not in {"baseline", "preserve"}:
            raise ValueError("method must be baseline or preserve")
        for name in (
            "n",
            "seed",
            "eval_every",
            "checkpoint_every",
            "opd_steps",
            "rollout_max_new_tokens",
            "kl_chunk_size",
            "unstructured_max_new_tokens",
            "multihop_max_new_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"Invalid integer {name}")
        if self.seed >= 2**32:
            raise ValueError("seed must be below 2**32")
        for name in ("opd_lr", "kl_weight"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Invalid {name}")
        if self.opd_lr == 0 or self.dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Invalid learning rate or dtype")
        if self.method == "preserve" and not self.reasoning_path:
            raise ValueError("OPD requires reasoning_path")
        if self.dataset == "unke" and not self.similarity_model_path:
            raise ValueError("UnKE requires a local MiniLM similarity_model_path")
        if self.hparams.get("batch_size", 1) != 1 or self.hparams.get("adapter_mode") != "value":
            raise ValueError("Only batch size 1 and value-mode HoReN are supported")
        if self.hparams.get("replacement") != "replace_last":
            raise ValueError("Only replace_last is supported")
        if type(self.hparams.get("n_iter")) is not int or self.hparams["n_iter"] < 1:
            raise ValueError("n_iter must be positive")
        if not math.isfinite(self.hparams["edit_lr"]) or self.hparams["edit_lr"] <= 0:
            raise ValueError("edit_lr must be positive")
        if len(self.hparams.get("inner_params", [])) != 1:
            raise ValueError("Exactly one edit layer is required")
        return self

    @property
    def evaluation_checkpoints(self):
        if self.dataset == "mquake":
            return [self.n]
        if self.dataset == "unke":
            points = {1, 10, 30, 100, 120, 500} if self.n <= 1000 else {5000}
            period = 1000 if self.n < 4000 else 2000
            points.update(range(period, self.n + 1, period))
        else:
            points = set(range(self.eval_every, self.n + 1, self.eval_every))
        return sorted(p for p in points | {self.n} if 1 <= p <= self.n)

    def engine_config(self):
        names = (
            "opd_steps",
            "opd_lr",
            "kl_weight",
            "rollout_max_new_tokens",
            "kl_chunk_size",
            "seed",
            "dataset",
        )
        return {**{k: getattr(self, k) for k in names}, "max_length": self.hparams.get("max_length", 512)}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path, **overrides):
        data = json.loads(Path(path).read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data).validate()
