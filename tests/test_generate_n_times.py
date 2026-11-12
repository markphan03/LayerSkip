"""Offline CLI coverage using a tiny real Llama model and exit classifier."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from generate_n_times import process_cli_arguments


@pytest.fixture
def local_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from train_ee_cnn import EarlyExitGRU

    torch.manual_seed(42)
    model_dir = tmp_path / "llama"
    vocab = {word: i for i, word in enumerate([
        "<unk>", "<s>", "What", "is", "the", "capital", "of", "France", "?", "Paris",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>")
    tokenizer.save_pretrained(model_dir)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=None,
    ))
    model.save_pretrained(model_dir)
    classifier = EarlyExitGRU(16, 4)
    checkpoint = tmp_path / "exit.pt"
    torch.save({"model_config": classifier.config, "model_state_dict": classifier.state_dict()}, checkpoint)
    return model_dir, checkpoint


@pytest.mark.parametrize("strategy", ["early_exit_speculative", "self_speculative", "autoregressive"])
def test_repeated_cli_generations(local_model, tmp_path, strategy):
    model_dir, checkpoint = local_model
    output = tmp_path / "reports" / "runs.json"
    command = [
        sys.executable, "generate_n_times.py", "--model", str(model_dir),
        "--generation_strategy", strategy, "--prompt", "What is the capital of France?",
        "--N", "6", "--warmup", "1", "--max_steps", "4", "--sample", "True",
        "--seed", "123", "--num_speculations", "2", "--output", str(output),
    ]
    if strategy == "early_exit_speculative":
        command += ["--ee_checkpoint", str(checkpoint), "--ee_threshold", "0.8",
                    "--ee_initial_fraction", "0.25", "--ee_gpu_branching", "auto"]
    elif strategy == "self_speculative":
        # Fixed-depth mode must not try to load a classifier.
        command += ["--exit_layer", "1", "--ee_checkpoint", str(tmp_path / "missing.pt")]
    env = {k: v for k, v in os.environ.items() if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def run():
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1], env=env,
            text=True, capture_output=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count("=== Run") == 6
        report = json.loads(output.read_text())
        sections = result.stdout.split("=== Run")[1:]
        for row, section in zip(report["runs"], sections):
            body = section.split("===\n", 1)[1]
            if row["run"] < 5:
                assert body.startswith(row["decoded_prediction"] + "\nTime:")
            else:
                assert body.startswith("Time:")
            assert f"Time per token: {row['time_per_token']:.6f}s/token" in body
        assert "Time per token:" in result.stdout.split("Completed 6 runs", 1)[1]
        return report

    report = run()
    assert len(report["runs"]) == 6
    assert [row["seed"] for row in report["runs"]] == list(range(123, 129))
    assert all(row["num_tokens_generated"] == 4 for row in report["runs"])
    assert report["summary"]["total_tokens"] == 24
    total_time = sum(row["total_time"] for row in report["runs"])
    assert report["summary"]["tokens_per_second"] == pytest.approx(24 / total_time)
    assert report["summary"]["time_per_token"] == pytest.approx(total_time / 24)
    acceptance = report["summary"]["mean_acceptance_rate"]
    if strategy == "autoregressive":
        assert acceptance is None
    else:
        assert 0 <= acceptance <= 1
    # Removing warmup must preserve each run's sampled output and fresh context.
    command[command.index("--warmup") + 1] = "0"
    repeated = run()
    assert [row["generation_strategy_result"]["predicted_tokens"] for row in report["runs"]] == [
        row["generation_strategy_result"]["predicted_tokens"] for row in repeated["runs"]
    ]


@pytest.mark.parametrize("extra", [
    ["--N", "0"], ["--warmup", "-1"], ["--prompt", " "],
    ["--generation_strategy", "unknown"],
    ["--generation_strategy", "self_speculative", "--num_speculations", "6"],
    ["--generation_strategy", "early_exit_speculative", "--num_speculations", "6", "--ee_threshold", "1.1"],
])
def test_invalid_arguments_rejected_before_loading(extra):
    with pytest.raises(SystemExit) as error:
        process_cli_arguments(["--model", "unused", "--prompt", "hello", *extra])
    assert error.value.code == 2
