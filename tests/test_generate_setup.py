"""Plain-Python startup and compatibility with torchrun rank selection."""

import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch

from arguments import Arguments
from generate import setup


LAUNCH_VARIABLES = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


def test_plain_python_setup_needs_no_process_group_and_preserves_seeding(monkeypatch):
    for name in LAUNCH_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    def unexpected_init(*args, **kwargs):
        pytest.fail("Single-process generation must not initialize a process group.")

    monkeypatch.setattr(torch.distributed, "init_process_group", unexpected_init)
    args = Arguments(model="unused", seed=123)
    setup(args, device="cpu")
    first_random, first_tensor = random.random(), torch.rand(3)
    setup(args, device="cpu")
    assert random.random() == first_random
    torch.testing.assert_close(torch.rand(3), first_tensor)


@pytest.mark.parametrize("global_rank,local_rank,should_exit", [
    ("0", "0", False),
    ("1", "1", True),
    ("2", "0", True),  # Local rank 0 on a second node must not generate.
    (None, "1", True),
])
def test_torchrun_rank_selection_without_rendezvous(monkeypatch, global_rank, local_rank, should_exit):
    for name in LAUNCH_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    if global_rank is not None:
        monkeypatch.setenv("RANK", global_rank)
    monkeypatch.setenv("LOCAL_RANK", local_rank)
    if should_exit:
        with pytest.raises(SystemExit) as stopped:
            setup(Arguments(model="unused"), device="cpu")
        assert stopped.value.code == 0
    else:
        setup(Arguments(model="unused"), device="cpu")


def test_python_command_generates_with_local_model_and_checkpoint(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from train_ee_cnn import EarlyExitGRU

    torch.manual_seed(42)
    model_dir = tmp_path / "llama"
    vocab = {word: index for index, word in enumerate([
        "<unk>", "<s>", "</s>", "Question", ":", "Why", "?", "Answer",
        "This", "is", "a", "warmup", "prompt", "the", "sky", "blue",
    ])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(model_dir)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
    ))
    model.save_pretrained(model_dir)
    classifier = EarlyExitGRU(16, 4)
    checkpoint = tmp_path / "exit.pt"
    torch.save({
        "model_config": classifier.config,
        "model_state_dict": classifier.state_dict(),
    }, checkpoint)
    env = {name: value for name, value in os.environ.items() if name not in LAUNCH_VARIABLES}
    env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [
            sys.executable, "generate.py", "--model", str(model_dir),
            "--generation_strategy", "early_exit_speculative", "--ee_checkpoint", str(checkpoint),
            "--ee_threshold", "0.5", "--num_speculations", "2", "--max_steps", "4",
            "--sample", "False", "--streamer", "none",
        ],
        cwd=Path(__file__).resolve().parents[1], env=env,
        input="Question: Why?\nAnswer:", text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Number of tokens:" in result.stdout
    assert "Acceptance Rate:" in result.stdout
