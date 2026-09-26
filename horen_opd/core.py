"""HoReN value memory with explicit edit-only slot allocation."""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn import functional as F


def resolve_layer(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    name = name.replace("[", ".").replace("]", "")
    if name.endswith((".weight", ".bias")):
        name = name.rsplit(".", 1)[0]
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


class HopfieldValueAdapter(nn.Module):
    """Value-mode Hopfield retrieval and residual updates."""

    def __init__(self, layer: nn.Linear, hparams: dict, config: dict):
        super().__init__()
        if not isinstance(layer, nn.Linear):
            raise TypeError("The OPD backend supports torch.nn.Linear down_proj only")
        if hparams.get("adapter_mode", "value") != "value":
            raise ValueError("HoReN-OPD requires adapter_mode=value")
        self.layer = layer
        self.hparams = dict(hparams)
        self.key_id = -1
        self.chosen_key = torch.tensor([0], device=layer.weight.device)
        self.allow_growth = False
        self.codebook_enabled = True
        self.edit_label: torch.Tensor | None = None
        key = torch.randn(1, layer.in_features, device=layer.weight.device, dtype=layer.weight.dtype)
        if self.hparams.get("normalize_codebook_keys", False):
            key = F.normalize(key, p=2, dim=-1)
        self.register_buffer("keys", key)
        self.values = nn.Parameter(
            torch.zeros(1, layer.out_features, device=layer.weight.device), requires_grad=False
        )
        self.key_labels = [torch.tensor(-1, device=layer.weight.device)]

        self.layer.requires_grad_(False)

    @property
    def weight(self):
        return self.layer.weight

    def _select_query(self, x: torch.Tensor, last: int) -> torch.Tensor:
        strategy = self.hparams.get("query_selection_strategy", "last_prompt_token")
        if strategy == "last_prompt_token" or last == -1:
            return x[:, last, :]
        if strategy == "first_prompt_token":
            return x[:, 0, :]
        length = last + 1
        percentage = re.fullmatch(r"last_(\d+(?:\.\d+)?)_perc_prompt_tokens_avg", strategy)
        last_n = re.fullmatch(r"last_(\d+)_prompt_tokens_avg", strategy)
        if percentage:
            fraction = float(percentage.group(1)) / 100
            if not 0 < fraction <= 1:
                raise ValueError(f"Invalid query strategy: {strategy}")
            count = max(math.ceil(length * fraction), 1)
        elif last_n and int(last_n.group(1)) > 0:
            count = min(int(last_n.group(1)), length)
        else:
            raise ValueError(f"Invalid query strategy: {strategy}")
        return x[:, length - count : length, :].mean(dim=1)

    def _query(self, query: torch.Tensor) -> torch.Tensor:
        keys = self.keys.to(device=query.device, dtype=query.dtype)
        qt = query
        for _ in range(int(self.hparams.get("hopfield_retrieval_max_iter", 8))):
            scores = float(self.hparams.get("hopfield_retrieval_beta", 1)) * (qt @ keys.t())
            retrieved = F.softmax(scores, dim=-1) @ keys
            if self.hparams.get("normalize_codebook_keys", False):
                retrieved = F.normalize(retrieved, p=2, dim=-1)
            if (retrieved - qt).norm(dim=-1).max().item() < float(
                self.hparams.get("hopfield_retrieval_eps", 1e-5)
            ):
                break
            alpha = float(self.hparams.get("hopfield_retrieval_alpha", 1))
            qt = (1 - alpha) * qt + alpha * retrieved
            if self.hparams.get("normalize_codebook_keys", False):
                qt = F.normalize(qt, p=2, dim=-1)
        return qt @ keys.t()

    @staticmethod
    def label_match(edit_label: torch.Tensor, key_label: torch.Tensor) -> bool:
        # Match the upstream mean-label rule, including its possible collisions.
        if key_label.numel() == 1 and key_label.item() == -1:
            return False
        return bool(edit_label.float().mean() == key_label.float().mean())

    def _add_key(self, query: torch.Tensor, output: torch.Tensor, index: int):
        if not self.allow_growth or self.edit_label is None:
            raise RuntimeError("Slot allocation requires an explicit edit label and growth capability")
        self.keys = torch.cat((self.keys, query.detach().to(self.keys.dtype)), dim=0)
        if self.hparams.get("val_init", "warm") == "cold":
            value = torch.rand(1, self.layer.out_features, device=self.weight.device, dtype=torch.float32)
        else:
            value = output[:, index, :].detach().float()
        self.values = nn.Parameter(torch.cat((self.values.detach(), value)), requires_grad=True)
        self.key_labels.append(self.edit_label.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layer(x)
        if not self.codebook_enabled or (len(self.keys) == 1 and not self.allow_growth):
            return output
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("HoReN-OPD v1 requires batch size 1 and [B,T,D] activations")
        index = min(self.key_id, x.shape[1] - 1)
        query = self._select_query(x, index)
        if self.hparams.get("normalize_codebook_keys", False):
            query = F.normalize(query, p=2, dim=-1)
        scores = self._query(query)
        threshold = float(self.hparams.get("hopfield_key_match_threshold", 0.95))
        if self.allow_growth:
            try:
                maximum, chosen = scores.max(dim=-1)
                if self.edit_label is None:
                    raise RuntimeError("Growth was enabled without a current edit label")
                if maximum.item() <= threshold or not self.label_match(
                    self.edit_label, self.key_labels[int(chosen.item())]
                ):
                    self._add_key(query, output, index)
                    scores = self._query(query)
            finally:
                self.allow_growth = False
        maximum, self.chosen_key = scores.max(dim=-1)
        update = self.values[self.chosen_key] * (maximum > threshold).unsqueeze(1).to(self.values.dtype)
        output = output.clone()
        output[:, index, :] += update.to(output.dtype)
        return output

    def snapshot(self) -> dict[str, Any]:
        cpu = lambda x: x.detach().cpu().clone()
        return {
            "keys": cpu(self.keys),
            "values": cpu(self.values),
            "key_labels": [cpu(label) for label in self.key_labels],
            "key_id": self.key_id,
            "chosen_key": cpu(self.chosen_key),
            "codebook_enabled": self.codebook_enabled,
        }

    def restore(self, state: dict):
        device = self.weight.device
        if state["keys"].shape[1] != self.layer.in_features or state["values"].shape != (
            len(state["keys"]),
            self.layer.out_features,
        ):
            raise ValueError("Malformed codebook checkpoint")
        if len(state["key_labels"]) != len(state["keys"]):
            raise ValueError("Codebook label count does not match key count")
        self.keys = state["keys"].to(device=device, dtype=self.weight.dtype).clone()
        self.values = nn.Parameter(
            state["values"].to(device=device, dtype=torch.float32).clone(), requires_grad=False
        )
        self.key_labels = [label.to(device).clone() for label in state["key_labels"]]
        self.key_id = int(state["key_id"])
        self.chosen_key = state["chosen_key"].to(device).clone()
        self.codebook_enabled = bool(state["codebook_enabled"])
        self.allow_growth = False
        self.edit_label = None

    @contextmanager
    def frozen_role(self, state: dict) -> Iterator[None]:
        """Swap only adapter tensors, retaining live student parameter identities."""
        names = (
            "keys",
            "values",
            "key_labels",
            "key_id",
            "chosen_key",
            "codebook_enabled",
            "allow_growth",
            "edit_label",
        )
        live = {name: getattr(self, name) for name in names}
        try:
            self.restore(state)
            with torch.no_grad():
                yield
        finally:
            for name, value in live.items():
                setattr(self, name, value)
