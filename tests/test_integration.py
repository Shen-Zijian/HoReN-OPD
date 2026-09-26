import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from horen_opd.checkpoint import sha256_file
from horen_opd.config import RunConfig, baseline_hparams
from horen_opd.data import BBH_TASKS, prepare_reasoning, save_reasoning
from horen_opd.run import run
from test_engine import hparams, make_model


@pytest.mark.parametrize("method", ["baseline", "preserve"])
def test_local_model_full_run_manifest_and_resume(tmp_path, method, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model_dir = tmp_path / "model"
        torch.manual_seed(7)
        make_model().save_pretrained(model_dir)
        vocabulary = {
            s: i
            for i, s in enumerate(
                [
                    "[PAD]",
                    "[EOS]",
                    "[BOS]",
                    "[UNK]",
                    "Alice",
                    "Bob",
                    "works",
                    "at",
                    "Paris",
                    "Rome",
                    "Where",
                    "does",
                    "live",
                    "London",
                    "?",
                ]
            )
        }
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
            bos_token="[BOS]",
        ).save_pretrained(model_dir)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sources = {
            "zsre_edit_data.json": [
                {
                    "src": f"{name} works at",
                    "subject": name,
                    "answers": [target],
                    "rephrase": f"Where does {name} live ?",
                }
                for name, target in (("Alice", "Paris"), ("Bob", "Rome"))
            ],
            "zsre_train_data.json": [{"loc": "Where does Bob live ?", "loc_ans": "London"}] * 2,
        }
        for name, rows in sources.items():
            (data_dir / name).write_text(json.dumps(rows))
        pool = prepare_reasoning(
            [{"question": f"train {i}", "answer": "#### 1"} for i in range(4)],
            [{"question": "test only", "answer": "#### 1"}],
            {
                task: [{"input": f"{task} problem {i}", "target": "(A)"} for i in range(5)]
                for task in BBH_TASKS
            },
            gsm_train_size=2,
            gsm_dev_size=1,
        )
        pool["manifest"].update(
            editing_dataset="zsre",
            editing_n_excluded=2,
            editing_inputs_sha256={name: sha256_file(data_dir / name) for name in sources},
        )
        pool_path = tmp_path / "pool.json"
        save_reasoning(pool, pool_path)
        config = RunConfig(
            model_path=str(model_dir),
            model_revision="random-test-model",
            data_path=str(data_dir),
            reasoning_path=str(pool_path),
            output_dir=str(tmp_path / "runs"),
            method=method,
            n=2,
            device="cpu",
            dtype="float32",
            eval_every=1,
            rollout_max_new_tokens=2,
            hparams={**baseline_hparams(), **hparams()},
        )
        assert run(config, check=True)["training_facts"] == 2
        result = run(config)
        assert result["n_training_facts"] == 2
        directory = tmp_path / "runs" / f"zsre_{method}_n2_s42"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        for name, record in manifest["artifacts"].items():
            assert sha256_file(directory / name) == record["sha256"]
        with pytest.raises(FileExistsError):
            run(config)
        monkeypatch.setattr(
            "horen_opd.run.initialize_engine",
            lambda _: pytest.fail("Completed resume must not allocate a model"),
        )
        assert run(config, resume=True) == result
        (directory / "edit_logs.json").write_text("[]")
        with pytest.raises(ValueError, match="artifacts changed"):
            run(config, resume=True)
    finally:
        torch.set_num_threads(previous_threads)
