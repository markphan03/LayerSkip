"""Exit scheduling, classifier precision, and real CUDA conditional execution."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from self_speculation.early_exit_speculation_generator import (
    EarlyExitSelfSpeculativeGenerationStrategy, _AdaptiveDraftCache, exit_check_layers,
)
from self_speculation.early_exit_cuda_graphs import ConditionalDraftGraphs, conditional_graphs_available
from self_speculation.generator_base import GenerationConfig
from train_ee_cnn import EarlyExitGRU


@pytest.mark.parametrize("layers,start,expected", [
    (32, 1, list(range(1, 33))),
    (35, 9, list(range(9, 36))),
    (8, 4, list(range(4, 9))),
    (8, 8, [8]),
    (1, 1, [1]),
])
def test_layer_by_layer_schedule(layers, start, expected):
    assert exit_check_layers(layers, classifier_start_layer=start) == expected


def test_cap_preserves_start_layer_or_forces_exit_before_it():
    assert exit_check_layers(35, 16, 9) == list(range(9, 17))
    assert exit_check_layers(35, 8, 9) == [8]


@pytest.mark.parametrize("start", [0, -1, 33, 1.5, True, float("nan"), float("inf")])
def test_invalid_start_layer(start):
    with pytest.raises(ValueError, match="ee-classifier-start-layer"):
        exit_check_layers(32, classifier_start_layer=start)


def tiny_model(attention="sdpa", device="cpu", dtype=torch.float32, layers=8):
    torch.manual_seed(41)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, eos_token_id=None,
    )
    config._attn_implementation = attention
    return LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def test_delayed_checks_consume_all_gru_states_once_and_reset_per_token():
    model = tiny_model()
    classifier = EarlyExitGRU(16, 4, num_layers=2).eval()
    classifier.head.weight.zero_()
    classifier.head.bias.fill_(-10)
    runner = _AdaptiveDraftCache(model, DynamicCache(), torch.tensor([[1, 5, 9]]))
    seen = []
    hook = classifier.gru.register_forward_pre_hook(
        lambda module, args: seen.append((args[0].shape[1], args[1] is None))
    )
    try:
        runner.draft_logits(classifier, .5, 8, 4)
        runner.append(3)
        runner.draft_logits(classifier, .5, 8, 4)
    finally:
        hook.remove()
    assert seen == ([(4, True)] + [(1, False)] * 3) * 2


@pytest.mark.parametrize("start,cap,expected_depth,expected_calls", [
    (1, 8, 1, 1), (4, 8, 4, 1), (4, 4, 4, 0), (6, 4, 4, 0),
])
@torch.inference_mode()
def test_first_exit_respects_start_layer_and_cap(start, cap, expected_depth, expected_calls):
    model = tiny_model()
    classifier = EarlyExitGRU(16, 4).eval()
    runner = _AdaptiveDraftCache(model, DynamicCache(), torch.tensor([[1, 5, 9]]))
    _, depth = runner.draft_logits(classifier, 0, cap, start)
    assert depth == expected_depth
    assert runner.counters.total_classifier_calls == expected_calls


def test_start_beyond_model_depth_fails_on_portable_path():
    from self_speculation.early_exit_speculation_generator import VerifiedSpeculativeGenerationStrategy

    strategy = VerifiedSpeculativeGenerationStrategy(
        EarlyExitGRU(16, 4), classifier_start_layer=9, gpu_branching="off",
    )
    with pytest.raises(ValueError, match="exceeds the model's number of layers"):
        strategy.prepare_for_generation(tiny_model(), [1, 5, 9], GenerationConfig())


@torch.inference_mode()
def test_grouped_gru_matches_training_scores_at_scheduled_layers():
    model = tiny_model(layers=35)
    classifier = EarlyExitGRU(16, 4, num_layers=2).eval()
    ids = torch.tensor([[1, 5, 9]])
    runner = _AdaptiveDraftCache(model, DynamicCache(), ids)
    checks = exit_check_layers(35, classifier_start_layer=9)
    for next_token in [3, 7, 2]:
        raw = []
        hooks = [layer.register_forward_hook(
            lambda module, args, output: raw.append(output[0][:, -1:])
        ) for layer in model.model.layers]
        try:
            model(ids, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        scores = classifier(torch.stack(raw, dim=1))[0, :, 0].sigmoid()
        threshold = scores[torch.tensor(checks[:-1]) - 1].mean().item()
        expected_depth = next((depth for depth in checks[:-1] if scores[depth - 1] >= threshold), 35)
        logits, depth = runner.draft_logits(classifier, threshold, 35, 9)
        assert depth == expected_depth
        expected_logits = model.lm_head(model.model.norm(raw[depth - 1]))[:, 0]
        torch.testing.assert_close(logits, expected_logits)
        runner.append(next_token)
        ids = torch.cat([ids, ids.new_tensor([[next_token]])], dim=1)


@pytest.mark.parametrize("repeat", [False, True])
def test_cli_start_layer_default_validation_and_removed_fraction(repeat):
    from generate import process_cli_arguments as parse_generate
    from generate_n_times import process_cli_arguments as parse_repeat

    parse = parse_repeat if repeat else parse_generate
    arguments = ["--model", "unused", "--generation_strategy", "early_exit_speculative",
                 "--num_speculations", "6"]
    if repeat:
        arguments += ["--prompt", "hello"]
    assert parse(arguments)[1].ee_classifier_start_layer == 1
    options = parse(arguments + ["--ee-classifier-start-layer", "4", "--ee_gpu_branching", "off"])[1]
    assert options.ee_classifier_start_layer == 4 and options.ee_gpu_branching == "off"
    for invalid in ["0", "-1", "1.5"]:
        with pytest.raises(SystemExit):
            parse(arguments + ["--ee-classifier-start-layer", invalid])
    with pytest.raises(ValueError, match="not used"):
        parse(arguments + ["--ee_initial_fraction", "0.25"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_classifier_matches_model_precision_before_generation(dtype, tmp_path):
    model = tiny_model(dtype=dtype)
    classifier = EarlyExitGRU(16, 4)
    path = tmp_path / "classifier.pt"
    torch.save({"model_config": classifier.config, "model_state_dict": classifier.state_dict()}, path)
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, gpu_branching="off")
    strategy.prepare_model(model)
    assert all(p.dtype == dtype and p.device == model.device for p in strategy.classifier.parameters())


CUDA_CONDITIONAL = torch.cuda.is_available() and hasattr(torch.cuda.CUDAGraph, "begin_capture_to_if_node")


@pytest.mark.skipif(not CUDA_CONDITIONAL, reason="Requires CUDA conditional graphs")
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@torch.inference_mode()
def test_cuda_graphs_skip_layers_and_preserve_cache_across_depth_changes_and_rejection(attention):
    model = tiny_model(attention, "cuda", torch.float16)
    classifier = EarlyExitGRU(16, 4, num_layers=2).cuda().half().eval()
    classifier.head.weight.zero_()
    checks = exit_check_layers(8, classifier_start_layer=4)
    engine = ConditionalDraftGraphs(model, classifier, checks, .5, 32, 3)
    ids = torch.tensor([[1, 5, 9]], device="cuda")
    engine.prefill(ids)
    cache = DynamicCache()
    for round_index in range(2):
        engine.begin_round(ids)
        eager = _AdaptiveDraftCache(model, cache, ids[:, cache.get_seq_length():])
        full = ids.clone()
        for index, (bias, token) in enumerate([(10, 3), (-10, 7), (10, 2)]):
            classifier.head.bias.fill_(bias)
            expected, expected_depth = eager.draft_logits(classifier, .5, 8, 4)
            actual, depth = engine.draft_logits(classifier, .5, 8, 4)
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            assert depth.item() == expected_depth == (4 if bias > 0 else 8)
            if index == 0:
                assert engine.completed.tolist() == [1] * 4 + [0] * 4
            elif index == 2:
                assert engine.completed.tolist() == [3] * 4 + [2] * 4
            eager.append(token)
            engine.append(token)
            full = torch.cat([full, full.new_tensor([[token]])], dim=1)
        actual = engine.verification_logits(4)
        expected = eager.verification_logits(4)
        baseline = model(full, use_cache=True)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(actual, baseline.logits[:, -4:], atol=2e-3, rtol=2e-3)
        for layer in range(8):
            for graph_cache, baseline_cache in zip(
                (engine.cache.key_cache[layer], engine.cache.value_cache[layer]),
                baseline.past_key_values[layer],
            ):
                torch.testing.assert_close(graph_cache[:, :, :full.shape[1]], baseline_cache, atol=2e-3, rtol=2e-3)
        # Accept one draft and emit a correction; overwrite discarded suffix.
        ids = torch.cat([ids, ids.new_tensor([[3, 11]])], dim=1)
        cache.crop(ids.shape[1] - 1)


@pytest.mark.skipif(not CUDA_CONDITIONAL, reason="Requires CUDA conditional graphs")
@torch.inference_mode()
def test_cuda_exit_replay_has_no_host_scalar_reads():
    model = tiny_model(device="cuda", dtype=torch.float16)
    classifier = EarlyExitGRU(16, 4).cuda().half().eval()
    engine = ConditionalDraftGraphs(model, classifier, exit_check_layers(8), .5, 32, 2)
    engine.begin_round(torch.tensor([[1]], device="cuda"))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        engine.draft_logits(classifier, .5, 8)
        torch.cuda.synchronize()
    assert not any(event.key in ("aten::item", "aten::_local_scalar_dense") for event in profile.key_averages())


@pytest.mark.skipif(not CUDA_CONDITIONAL, reason="Requires CUDA conditional graphs")
@torch.inference_mode()
def test_cuda_strategy_greedy_and_sampled_rejection_match_target(tmp_path):
    model = tiny_model(device="cuda", dtype=torch.float16)
    classifier = EarlyExitGRU(16, 4)
    path = tmp_path / "classifier.pt"
    torch.save({"model_config": classifier.config, "model_state_dict": classifier.state_dict()}, path)
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold=0, gpu_branching="on")
    prompt = [1, 5, 9]
    baseline = model.generate(
        torch.tensor([prompt], device="cuda"), max_new_tokens=12, do_sample=False, pad_token_id=0,
    )[0, len(prompt):].tolist()
    for sample in [False, True]:
        result = strategy.generate_token_ids(
            model, prompt, [], GenerationConfig(max_steps=12, num_speculations=3, sample=sample, top_k=1),
        )
        assert result.predicted_tokens == baseline
        assert result.acceptance_rate < 1
        assert conditional_graphs_available(model)
        assert strategy._cuda_graphs is not None
