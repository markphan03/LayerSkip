"""Per-layer checks with a hard draft-depth cap at layer 10."""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from self_speculation.early_exit_cuda_graphs import conditional_graphs_available
from self_speculation.early_exit_speculation_generator import (
    VerifiedSpeculativeGenerationStrategy, _AdaptiveDraftCache, exit_check_layers,
)
from self_speculation.generator_base import GenerationConfig
from tests.test_early_exit_latency import tiny_model
from train_ee_cnn import EarlyExitGRU


def test_schedule_checks_each_layer_up_to_cap():
    assert exit_check_layers(32, max_depth=10, classifier_start_layer=1) == list(range(1, 11))
    assert exit_check_layers(32, classifier_start_layer=1)[:3] == [1, 2, 3]


@pytest.mark.parametrize("exit_depth,checks", [(1, 1), (9, 9), (10, 9)])
@torch.inference_mode()
def test_drafting_never_executes_beyond_layer_ten(exit_depth, checks):
    model = tiny_model(layers=32)
    classifier = EarlyExitGRU(16, 4).eval()
    ids = torch.tensor([[1, 5, 9]])
    raw = []
    handle = model.model.layers[exit_depth - 1].register_forward_hook(
        lambda module, args, output: raw.append(output[0][:, -1:])
    )
    model(ids)
    handle.remove()
    expected = model.lm_head(model.model.norm(raw[0]))[:, 0]
    visited, classified = [], []

    def score(module, args, output):
        classified.append(visited[-1])
        return torch.full_like(output, 10 if exit_depth != 10 and visited[-1] == exit_depth else -10)

    handles = [layer.register_forward_pre_hook(
        lambda module, args, depth=index + 1: visited.append(depth)
    ) for index, layer in enumerate(model.model.layers)]
    handles.append(classifier.head.register_forward_hook(score))
    try:
        runner = _AdaptiveDraftCache(model, DynamicCache(), ids)
        logits, depth = runner.draft_logits(classifier, .8, 10, classifier_start_layer=1)
    finally:
        for handle in handles:
            handle.remove()
    assert depth == exit_depth
    assert visited == list(range(1, exit_depth + 1))
    assert classified == list(range(1, checks + 1))
    assert runner.counters.total_classifier_calls == checks
    torch.testing.assert_close(logits, expected)


@pytest.mark.parametrize("mode", ["off", "auto", "on"])
@pytest.mark.parametrize("threshold", [0, 1])
@torch.inference_mode()
def test_verified_generation_and_graphs_follow_policy(mode, threshold):
    if mode != "off" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    model = tiny_model(layers=32, device="cpu" if mode == "off" else "cuda",
                       dtype=torch.float32 if mode == "off" else torch.float16)
    if mode == "on" and not conditional_graphs_available(model):
        pytest.skip("Requires conditional graph API")
    strategy = VerifiedSpeculativeGenerationStrategy(
        EarlyExitGRU(16, 4), threshold=threshold, classifier_start_layer=1,
        gpu_branching="off",
    )
    config = GenerationConfig(max_steps=6, num_speculations=2, sample=False, exit_layer=10)
    expected = strategy.generate_token_ids(model, [1, 5, 9], [], config)
    target = model.generate(torch.tensor([[1, 5, 9]], device=model.device),
                            max_new_tokens=6, do_sample=False, pad_token_id=0)[0, 3:].tolist()
    assert expected.predicted_tokens == target
    assert sum(expected.draft_exit_counts[10:]) == 0
    drafts = sum(expected.draft_exit_counts)
    assert expected.total_classifier_calls == drafts * (1 if threshold == 0 else 9)
    assert expected.draft_exit_counts[0 if threshold == 0 else 9] == drafts
    strategy.gpu_branching = mode
    for _ in range(2):
        actual = strategy.generate_token_ids(model, [1, 5, 9], [], config)
        assert actual == expected
        if mode != "off":
            assert strategy._cuda_graphs.checks == list(range(1, 11))


def test_cli_settings_supported_by_both_generation_commands():
    from generate import process_cli_arguments as parse_generate
    from generate_n_times import process_cli_arguments as parse_repeat
    arguments = ["--model", "unused", "--generation_strategy", "early_exit_speculative",
                 "--num_speculations", "6", "--exit_layer", "10", "--ee-classifier-start-layer", "1"]
    for parsed in (parse_generate(arguments), parse_repeat(arguments + ["--prompt", "hello"])):
        assert parsed[1].ee_classifier_start_layer == 1
        assert parsed[2].exit_layer == 10
