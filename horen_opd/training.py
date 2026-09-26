"""Sequential HoReN edits and student-on-policy reverse KL."""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .core import HopfieldValueAdapter, resolve_layer


def reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Full-vocabulary D_KL(student || teacher), mean over valid token positions."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Teacher/student logits must have matching [..., vocabulary] shapes")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.detach().reshape_as(student)
    valid = (
        torch.ones(student.shape[0], device=student.device, dtype=torch.bool)
        if mask is None
        else mask.reshape(-1).bool()
    )
    if valid.numel() != student.shape[0] or not bool(valid.any()):
        raise ValueError("KL mask must contain at least one valid token")
    total = student.reshape(-1)[0].float() * 0
    for start in range(0, len(student), chunk_size):
        sl = slice(start, start + chunk_size)
        log_s = F.log_softmax(student[sl].float(), dim=-1)
        log_t = F.log_softmax(teacher[sl].float(), dim=-1)
        per_token = (log_s.exp() * (log_s - log_t)).sum(dim=-1)
        total = total + per_token[valid[sl]].sum()
    result = total / valid.sum()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("Non-finite on-policy reverse KL")
    return result


class OpdEngine:
    def __init__(self, model, tokenizer, hparams: dict, config: dict):
        self.model, self.tokenizer = model, tokenizer
        self.hparams, self.config = dict(hparams), dict(config)
        self.model.eval().requires_grad_(False)
        layer_path = self.hparams["inner_params"][0]
        parent, name = resolve_layer(model, layer_path)
        self.adapter = HopfieldValueAdapter(getattr(parent, name), self.hparams, self.config)
        setattr(parent, name, self.adapter)
        self.model.eval()
        self.edit_index = 0
        self._rollout_cpu_rng = (
            torch.Generator(device="cpu").manual_seed(int(config.get("seed", 42)) + 2771).get_state()
        )
        self._rollout_device_rng = None
        if self.device.type == "cuda":
            self._rollout_device_rng = (
                torch.Generator(device=self.device)
                .manual_seed(int(config.get("seed", 42)) + 3771)
                .get_state()
            )
        elif self.device.type == "mps":
            old = torch.mps.get_rng_state()
            torch.mps.manual_seed(int(config.get("seed", 42)) + 3771)
            self._rollout_device_rng = torch.mps.get_rng_state()
            torch.mps.set_rng_state(old)

    @property
    def device(self):
        return self.adapter.weight.device

    @contextmanager
    def _sampling_rng(self):
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda"
            else []
        )
        mps_old = torch.mps.get_rng_state() if self.device.type == "mps" else None
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._rollout_cpu_rng)
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self._rollout_device_rng, self.device)
            elif self.device.type == "mps":
                torch.mps.set_rng_state(self._rollout_device_rng)
            try:
                yield
            finally:
                self._rollout_cpu_rng = torch.get_rng_state().clone()
                if self.device.type == "cuda":
                    self._rollout_device_rng = torch.cuda.get_rng_state(self.device).clone()
                elif self.device.type == "mps":
                    self._rollout_device_rng = torch.mps.get_rng_state().clone()
                    torch.mps.set_rng_state(mps_old)

    def _prompt_tokens(self, prompt: str, truncate: bool = False) -> dict:
        args = {"return_tensors": "pt"}
        if truncate:
            args.update(
                truncation=True,
                max_length=int(self.config.get("max_length", self.hparams.get("max_length", 512))),
            )
        tokens = self.tokenizer(prompt, **args)
        return {
            key: value.to(self.device)
            for key, value in tokens.items()
            if key in {"input_ids", "attention_mask"}
        }

    def _edit_tokens(self, request: dict) -> dict:
        prompt, target = request["prompt"], request["target_new"]
        if self.config.get("dataset", "zsre") == "unke" and not target.startswith(" "):
            # Preserve native UnKE's extra leading space before the shared separator.
            target = " " + target
        prompt_ids = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)["input_ids"]
        length = int((prompt_ids[0] != self.tokenizer.pad_token_id).sum())
        tokens = self.tokenizer([f"{prompt} {target}"], return_tensors="pt", padding=True, truncation=True)
        tokens = {
            key: value.to(self.device)
            for key, value in tokens.items()
            if key in {"input_ids", "attention_mask"}
        }
        labels = tokens["input_ids"].clone()
        labels[:, :length] = -100
        labels[tokens["input_ids"] == self.tokenizer.pad_token_id] = -100
        if not bool((labels[:, 1:] != -100).any()):
            raise ValueError("Edit target was empty or completely truncated")
        tokens["labels"] = labels
        return tokens

    def _forward(self, tokens: dict, boundary: int):
        self.adapter.key_id = boundary
        self.adapter.allow_growth = False
        return self.model(**tokens, use_cache=False)

    def _generate_ids(
        self,
        prompt: str,
        max_new_tokens: int,
        sample: bool,
        truncate: bool = False,
        generation_overrides: dict | None = None,
    ) -> tuple[torch.Tensor, int]:
        tokens = self._prompt_tokens(prompt, truncate=truncate)
        length = tokens["input_ids"].shape[1]
        capacity = getattr(self.model.config, "max_position_embeddings", None)
        if capacity is not None and length + int(max_new_tokens) > capacity:
            raise ValueError(
                f"Prompt ({length}) plus continuation ({max_new_tokens}) exceeds model context ({capacity}); no silent truncation is applied"
            )
        self.adapter.key_id = length - 1
        self.adapter.allow_growth = False
        kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=sample,
            use_cache=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if generation_overrides:
            kwargs.update(generation_overrides)
        with torch.no_grad():
            if sample:
                ids = tokens["input_ids"].clone()
                attention = tokens.get("attention_mask", torch.ones_like(ids)).clone()
                eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
                if eos is None:
                    eos = self.tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                with self._sampling_rng():
                    for _ in range(int(max_new_tokens)):
                        logits = (
                            self._forward({"input_ids": ids, "attention_mask": attention}, length - 1)
                            .logits[:, -1]
                            .float()
                        )
                        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                        ids = torch.cat((ids, next_id), dim=1)
                        attention = torch.cat((attention, torch.ones_like(next_id)), dim=1)
                        if int(next_id.item()) in eos_ids:
                            break
            else:
                ids = self.model.generate(
                    input_ids=tokens["input_ids"], attention_mask=tokens.get("attention_mask"), **kwargs
                )
        return ids, length

    def generate_tokens(self, prompt: str, max_new_tokens: int, sample: bool = False) -> list[int]:
        """Full prompt + continuation IDs, matching released evaluation behavior."""
        return self._generate_ids(prompt, max_new_tokens, sample)[0][0].tolist()

    def generate(self, prompt: str, max_new_tokens: int = 1024, sample: bool = False) -> str:
        ids, length = self._generate_ids(
            prompt,
            max_new_tokens,
            sample,
            generation_overrides={
                "repetition_penalty": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "temperature": 1.0,
                "num_beams": 1,
            },
        )
        return self.tokenizer.decode(ids[0, length:].tolist(), skip_special_tokens=True)

    def _kl_on_fresh_rollout(self, prompt: str, teacher: dict) -> torch.Tensor:
        ids, length = self._generate_ids(prompt, int(self.config.get("rollout_max_new_tokens", 256)), True)
        if ids.shape[1] <= length:
            raise RuntimeError("Student rollout produced no continuation tokens")

        scoring = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        with self.adapter.frozen_role(teacher):
            teacher_logits = self._forward(scoring, length - 1).logits[:, length - 1 : -1, :].detach()
        student_logits = self._forward(scoring, length - 1).logits[:, length - 1 : -1, :]
        return reverse_kl(
            student_logits, teacher_logits, chunk_size=int(self.config.get("kl_chunk_size", 16))
        )

    def _ce(self, request: dict):
        tokens = self._edit_tokens(request)
        boundary = int((tokens["labels"] == -100).sum(dim=1).min().item() - 1)
        loss = self._forward(tokens, boundary).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite edit loss")
        return loss

    def _joint_step(self, request: dict, prompts: list[str], teacher: dict, optimizer) -> dict:
        optimizer.zero_grad(set_to_none=True)
        kl_values = []
        weight = float(self.config.get("kl_weight", 1))
        for prompt in prompts:
            kl = self._kl_on_fresh_rollout(prompt, teacher)
            (weight * kl / len(prompts)).backward()
            kl_values.append(float(kl.detach()))
        ce = self._ce(request)
        ce.backward()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite OPD gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {
            "edit_loss": float(ce.detach()),
            "reverse_kl": sum(kl_values) / len(kl_values) if kl_values else 0.0,
            "rollouts": len(kl_values),
        }

    def edit(self, request: dict, preservation_prompts: list[str], preserve: bool = True) -> dict:
        preservation_steps = int(self.config.get("opd_steps", 1)) if preserve else 0
        teacher = self.adapter.snapshot() if preservation_steps > 0 else None
        self.model.eval().requires_grad_(False)
        self.adapter.values.requires_grad_(True)
        tokens = self._edit_tokens(request)
        boundary = int((tokens["labels"] == -100).sum(dim=1).min().item() - 1)
        self.adapter.key_id = boundary
        self.adapter.edit_label = tokens["labels"].detach().clone()
        self.adapter.allow_growth = True
        losses, optimizer, best, stale = [], None, float("inf"), 0
        reason = "max_iterations"
        started = time.monotonic()
        try:
            for index in range(int(self.hparams.get("n_iter", 20))):
                self.adapter.key_id = boundary

                outputs = self.model(**tokens, use_cache=False)
                if index == 0:
                    optimizer = torch.optim.Adam(
                        [self.adapter.values], lr=float(self.hparams.get("edit_lr", 0.01))
                    )
                loss = outputs.loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite supervised HoReN edit loss")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                value = float(loss.detach())
                losses.append(value)
                if self.hparams.get("early_stop_loss") is not None and value <= float(
                    self.hparams["early_stop_loss"]
                ):
                    reason = "loss_threshold"
                    break
                if value < best:
                    best, stale = value, 0
                else:
                    stale += 1
                    patience = int(self.hparams.get("early_stop_patience", 0) or 0)
                    if patience and stale >= patience:
                        reason = "no_improvement"
                        break
            if not losses:
                raise ValueError("n_iter must be at least one")
            supervised_chosen = int(self.adapter.chosen_key.item())
            self.adapter.allow_growth = False
            opd = []
            if preservation_steps > 0:
                if not preservation_prompts:
                    raise ValueError("Preservation needs at least one historical or reasoning prompt")
                optimizer = torch.optim.Adam([self.adapter.values], lr=float(self.config.get("opd_lr", 1e-3)))
                for _ in range(preservation_steps):
                    opd.append(self._joint_step(request, preservation_prompts, teacher, optimizer))
            self.edit_index += 1
            return {
                "edit_index": self.edit_index,
                "chosen_key": supervised_chosen,
                "nkeys": len(self.adapter.keys),
                "num_memory_slots": len(self.adapter.keys) - 1,
                "key_id": boundary,
                "n_steps": len(losses),
                "initial_loss": losses[0],
                "final_loss": losses[-1],
                "min_loss": min(losses),
                "stop_reason": reason,
                "losses": losses,
                "opd": opd,
                "elapsed_seconds": time.monotonic() - started,
            }
        finally:
            self.adapter.allow_growth = False
            self.adapter.edit_label = None
            self.model.requires_grad_(False)

    @contextmanager
    def original_context(self):
        flags = self.adapter.codebook_enabled, self.adapter.allow_growth, self.adapter.key_id
        self.adapter.codebook_enabled = False
        self.adapter.allow_growth = False
        try:
            yield
        finally:
            self.adapter.codebook_enabled, self.adapter.allow_growth, self.adapter.key_id = flags

    def snapshot(self) -> dict:
        return {
            "schema_version": 2,
            "adapter": self.adapter.snapshot(),
            "edit_index": self.edit_index,
            "rollout_cpu_rng": self._rollout_cpu_rng.clone(),
            "rollout_device_rng": None
            if self._rollout_device_rng is None
            else self._rollout_device_rng.clone(),
        }

    def restore(self, snapshot: dict):
        if snapshot.get("schema_version") != 2:
            raise ValueError("Unsupported engine checkpoint version")
        self.adapter.restore(snapshot["adapter"])
        self.edit_index = int(snapshot["edit_index"])
        self._rollout_cpu_rng = snapshot["rollout_cpu_rng"].clone()
        self._rollout_device_rng = (
            None if snapshot["rollout_device_rng"] is None else snapshot["rollout_device_rng"].clone()
        )
        self.model.eval().requires_grad_(False)

    def stats(self) -> dict:
        key_bytes = self.adapter.keys.numel() * self.adapter.keys.element_size()
        value_bytes = self.adapter.values.numel() * self.adapter.values.element_size()
        label_bytes = sum(label.numel() * label.element_size() for label in self.adapter.key_labels)
        memory = key_bytes + value_bytes + label_bytes
        return {
            "num_memory_slots": len(self.adapter.keys) - 1,
            "codebook_size": len(self.adapter.keys),
            "codebook_bytes": memory,
            "codebook_key_bytes": key_bytes,
            "codebook_value_bytes": value_bytes,
            "key_label_bytes": label_bytes,
            "storage_definition": "tensor_payload_bytes_including_dummy_slot",
            "adapter_parameters": self.adapter.values.numel(),
            "deployment": "codebook",
        }
