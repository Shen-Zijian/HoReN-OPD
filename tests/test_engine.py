import pytest
import torch
from horen_opd.training import OpdEngine, reverse_kl


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return ([2] if add_special_tokens else []) + [3 + ord(c) % 29 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) - 3) % 26) for i in ids if not skip_special_tokens or int(i) > 2)

    def __call__(self, texts, return_tensors="pt", padding=False, truncation=False, max_length=None):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [self.encode(text)[: max_length or 128] if truncation else self.encode(text) for text in texts]
        longest = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (longest - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (longest - len(row)) for row in rows]),
        }


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen2ForCausalLM(config).eval()


def hparams():
    return dict(
        inner_params=["model.layers[0].mlp.down_proj.weight"],
        adapter_mode="value",
        n_iter=2,
        edit_lr=0.2,
        val_init="cold",
        normalize_codebook_keys=True,
        query_selection_strategy="last_60_perc_prompt_tokens_avg",
        hopfield_retrieval_beta=20,
        hopfield_retrieval_alpha=0.1,
        hopfield_retrieval_max_iter=1,
        hopfield_retrieval_eps=1e-5,
        hopfield_key_match_threshold=0.55,
    )


def make_engine():
    torch.manual_seed(42)
    return OpdEngine(
        make_model(),
        TinyTokenizer(),
        hparams(),
        dict(seed=42, rollout_max_new_tokens=3, max_length=32, kl_chunk_size=2),
    )


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def backbone(engine):
    return {
        name: p.detach().clone()
        for name, p in engine.model.named_parameters()
        if not name.endswith(("values",))
    }


def test_reverse_kl_direction_mask_gradients_and_extremes():
    student = torch.tensor([[[2.0, -1.0, 0.5], [0.1, 1.0, -2.0]]], requires_grad=True)
    teacher = torch.tensor([[[-2.0, 1.0, 0.3], [1.0, 0.2, 0.4]]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    actual = reverse_kl(student, teacher, mask, chunk_size=1)
    log_s, log_t = student[0, 0].log_softmax(-1), teacher[0, 0].detach().log_softmax(-1)
    expected = (log_s.exp() * (log_s - log_t)).sum()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert teacher.grad is None
    assert student.grad[0, 0].abs().sum() > 0
    assert student.grad[0, 1].abs().sum() == 0
    extreme = torch.tensor([[[10000.0, -10000.0]]], requires_grad=True)
    loss = reverse_kl(extreme, -extreme.detach())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    with pytest.raises(ValueError):
        reverse_kl(student, teacher, torch.zeros_like(mask))


def test_non_edit_training_mode_never_allocates_slots():
    engine = make_engine()
    engine.adapter.train()
    engine.adapter.edit_label = torch.tensor([[7]])
    for _ in range(3):
        engine.model(**engine._prompt_tokens("xy"), use_cache=False)
    assert len(engine.adapter.keys) == 1
    engine.generate("xy", 2)
    assert len(engine.adapter.keys) == 1


def test_frozen_role_is_independent_and_restores_parameter_identity():
    engine = make_engine()
    state = engine.adapter.snapshot()
    live_a = engine.adapter.values
    with torch.no_grad():
        engine.adapter.values.add_(10)
    with engine.adapter.frozen_role(state):
        assert engine.adapter.values is not live_a
        assert not engine.adapter.values.requires_grad
        assert_nested_equal(engine.adapter.values.cpu(), state["values"])
    assert engine.adapter.values is live_a
    assert not torch.equal(live_a.cpu(), state["values"])


def test_generation_recompute_agree_and_sampling_does_not_shift_edit_rng():
    engine = make_engine()
    engine.edit({"prompt": "xy", "target_new": "z"}, [], preserve=False)
    ids, length = engine._generate_ids("xy", 3, False)
    logits = engine._forward({"input_ids": ids, "attention_mask": torch.ones_like(ids)}, length - 1).logits
    assert ids[0, length:].tolist() == logits[0, length - 1 : -1].argmax(-1).tolist()
    before = torch.get_rng_state().clone()
    engine._generate_ids("xy", 3, True)
    assert torch.equal(before, torch.get_rng_state())
    assert engine.adapter.key_id == length - 1


def test_reasoning_neutralizes_penalty_but_edit_generation_preserves_upstream(monkeypatch):
    engine = make_engine()
    engine.model.generation_config.repetition_penalty = 1.05
    called = []
    original = engine.model.generate

    def capture(*args, **kwargs):
        called.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(engine.model, "generate", capture)
    engine.generate_tokens("xy", 2)
    assert "repetition_penalty" not in called[-1]
    engine.generate("xy", 2)
    assert called[-1]["repetition_penalty"] == 1.0
    assert called[-1]["num_beams"] == 1 and called[-1]["do_sample"] is False
    assert engine.model.generation_config.repetition_penalty == 1.05
