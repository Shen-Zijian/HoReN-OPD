import random
import math
import pytest
from horen_opd import evaluation
from horen_opd.evaluation import UNSTRUCTURED_KEYS, _safe_bleu_like, evaluate_unstructured_edits


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens if token != 1)


class Engine:
    tokenizer = CharTokenizer()

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def generate_tokens(self, prompt, max_new_tokens, sample):
        self.calls.append((prompt, max_new_tokens, sample))
        return self.tokenizer.encode(prompt) + self.tokenizer.encode(
            self.answers[prompt], add_special_tokens=False
        )

    def generate(self, *args, **kwargs):
        raise AssertionError("UnKE must preserve native generation config via generate_tokens")


class FixedRouge:
    """Injectable scorer that makes within-case vs global probe weighting clear."""

    def get_scores(self, prediction, reference):
        score = 1.0 if prediction.strip() == reference.strip() else 0.0
        return [{key: {"r": score, "p": 0.123, "f": 0.456} for key in ("rouge-1", "rouge-2", "rouge-l")}]


def request(index=0, sub=None):
    return {
        "case_id": index,
        "prompt": f"chat-original-{index}",
        "rephrase_prompt": f"chat-para-{index}",
        "target_new": "the target answer<|im_end|>",
        "locality": {},
        "portability": {"sub": {"prompt": list(sub or []), "ground_truth": ["target"] * len(sub or [])}},
    }


def test_native_bleu_is_whitespace_bleu1_like_not_bleu4():
    assert _safe_bleu_like("a b c d", "a b") == pytest.approx(math.exp(-1))
    assert _safe_bleu_like("a b", "a a a") == pytest.approx(1 / 3)
    assert _safe_bleu_like("a b", "a b c") == pytest.approx(2 / 3)
    assert _safe_bleu_like("Answer", "answer") == 0
    assert _safe_bleu_like("", "a") == _safe_bleu_like("a", "") == 0


def test_native_rouge_uses_recall_not_f1():
    pytest.importorskip("rouge")
    scored = evaluation._rouge_recall("a b c d", "a b", evaluation._native_rouge())
    assert scored == pytest.approx({"rouge1": 0.5, "rouge2": 1 / 3, "rougeL": 0.5})
    assert evaluation._rouge_recall("answer", "", evaluation._native_rouge()) == {
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
    }
    assert evaluation._rouge_recall("answer", "...", evaluation._native_rouge())["rougeL"] == 0


def test_unke_native_fields_macro_sub_and_semantic_label():
    requests = [request(0, ["sub0", "sub1"]), request(1, ["sub2"])]
    answers = {f"chat-{name}-{i}": "the target answer" for name in ("original", "para") for i in (0, 1)}
    answers.update(sub0="target", sub1="wrong", sub2="target")
    engine = Engine(answers)
    captured = []

    def similarities(references, predictions):
        captured.append((list(references), list(predictions)))
        return [-0.1, 0.5, 1.0, 0.75]

    result = evaluate_unstructured_edits(
        engine, requests, "local-minilm", similarity_scorer=similarities, rouge_scorer=FixedRouge()
    )
    assert len(UNSTRUCTURED_KEYS) == 13
    assert set(result["metrics"]) == set(UNSTRUCTURED_KEYS) | {"locality"}
    assert result["metrics"]["original_bleu"] == result["metrics"]["para_rougeL"] == 1
    assert result["metrics"]["sub_rougeL"] == 0.75  # (mean(1, 0) + mean(1)) / 2, not 2 / 3
    assert result["metrics"]["original_semantic_similarity"] == 0.45
    assert result["metrics"]["para_semantic_similarity"] == 0.625
    assert result["metrics"]["locality"] is None
    assert result["locality_status"] == "not_applicable"
    assert result["original_locality"] == {}
    assert result["portability_status"] == "available"
    assert result["native_metrics"]["Original"]["Bert Score"] == 0.45
    assert result["native_metrics"]["Sub"] == {"ROUGE-1": 0.75, "ROUGE-2": 0.75, "ROUGE-L": 0.75}
    assert result["semantic_similarity_definition"]["is_bertscore"] is False
    assert captured[0][0] == ["the target answer"] * 4  # suffix absent during scoring
    assert len(engine.calls) == 7
    assert all(call[1:] == (512, False) for call in engine.calls)
    assert result["generation"]["use_cache"] is False
    assert result["individuals"][0]["requested_rewrite"] == requests[0]
    assert result["individuals"][0]["predictions"]["original"]["prediction"] == "the target answer"


def test_empty_predictions_zero_overlap_and_sub_na_not_fabricated():
    captured = []
    result = evaluate_unstructured_edits(
        Engine({"chat-original-0": "", "chat-para-0": "word"}),
        [request()],
        "unused",
        similarity_scorer=lambda ref, pred: captured.append(list(pred)) or [0.0, 0.3],
        rouge_scorer=FixedRouge(),
    )
    assert result["metrics"]["original_bleu"] == result["metrics"]["original_rougeL"] == 0
    assert result["metrics"]["sub_rougeL"] is None
    assert result["portability_status"] == "not_applicable"
    assert captured == [[" ", "word "]]  # exactly native single-token preprocessing


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda r: r.pop("case_id"), "case_id"),
        (lambda r: r.update(rephrase_prompt=None), "nonempty"),
        (lambda r: r.update(target_new="<|im_end|>"), "empty after"),
        (lambda r: r.update(locality={"invented": {}}), "no locality"),
        (lambda r: r.update(portability={"personas": {}}), "sub-question"),
        (lambda r: r.update(sub_question=["q"], sub_answer=[], portability={}), "count mismatch"),
    ],
)
def test_invalid_unke_evidence_rejected_before_generation(mutation, message):
    row = request()
    mutation(row)
    engine = Engine({})
    with pytest.raises(ValueError, match=message):
        evaluate_unstructured_edits(engine, [row], "unused", rouge_scorer=FixedRouge())
    assert engine.calls == []


def test_duplicate_ids_and_invalid_embedding_outputs_fail_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_unstructured_edits(Engine({}), [request(), request()], "unused", rouge_scorer=FixedRouge())
    for scores, message in [([0.5], "count mismatch"), ([math.nan, 0.5], "finite"), ([1.2, 0.3], "finite")]:
        with pytest.raises(ValueError, match=message):
            evaluate_unstructured_edits(
                Engine({"chat-original-0": "a", "chat-para-0": "a"}),
                [request()],
                "unused",
                similarity_scorer=lambda ref, pred: scores,
                rouge_scorer=FixedRouge(),
            )


def test_lazy_embedding_initialization_does_not_change_training_rng(monkeypatch, tmp_path):
    import numpy as np
    import torch
    from horen_opd.checkpoint import capture_rng

    class Model:
        def encode(self, texts, **kwargs):
            random.random()
            np.random.random()
            torch.rand(3)
            return torch.tensor([[float(len(text)), 1.0] for text in texts])

    def load(*args):
        random.random()
        np.random.random()
        torch.rand(2)
        return Model()

    monkeypatch.setattr(evaluation, "_local_sentence_model", load)
    before = capture_rng()
    values = evaluation._sentence_embedding_similarity(["a", "same"], ["bb", "same"], str(tmp_path))
    after = capture_rng()
    assert values[1] == pytest.approx(1.0)
    assert values[0] == pytest.approx(3 / math.sqrt(10))
    assert before["python"] == after["python"]
    assert before["numpy"][0] == after["numpy"][0]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert torch.equal(before["torch"], after["torch"])


def test_embedding_loader_refuses_nonlocal_hub_ids_before_import():
    with pytest.raises(FileNotFoundError, match="Local UnKE"):
        evaluation._local_sentence_model("missing/sentence-transformers-model", "cpu")
