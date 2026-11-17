"""Compare reported work with independently observed model/classifier calls."""

from dataclasses import asdict

import pytest
import torch

from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.early_exit_cuda_graphs import conditional_graphs_available
from self_speculation.early_exit_speculation_generator import VerifiedSpeculativeGenerationStrategy, _AdaptiveDraftCache
from self_speculation.generator_base import GenerationConfig, GenerationResult
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from tests.test_early_exit_latency import tiny_model
from train_ee_cnn import EarlyExitGRU


def strategy_for(kind, mode="off", threshold=0):
    if kind == "autoregressive":
        return AutoRegressiveGenerationStrategy()
    if kind == "fixed":
        return SelfSpeculativeGenerationStrategy(cuda_graphs=mode)
    return VerifiedSpeculativeGenerationStrategy(
        EarlyExitGRU(16, 4), threshold=threshold, classifier_start_layer=1, gpu_branching=mode,
    )


@pytest.mark.parametrize("kind", ["autoregressive", "fixed", "adaptive"])
@pytest.mark.parametrize("steps,drafts", [(0, 3), (1, 3), (7, 0), (7, 3)])
@pytest.mark.parametrize("prompt", [[1], [1, 5, 9]])
@torch.inference_mode()
def test_counts_match_real_calls_and_reset_between_requests(kind, steps, drafts, prompt, monkeypatch):
    model = tiny_model(layers=4)
    strategy = strategy_for(kind)
    config = GenerationConfig(max_steps=steps, num_speculations=drafts, exit_layer=2, sample=False)
    observed = dict(total_layer_executions=0, total_classifier_calls=0)
    observed_exits = []
    observed_phases = dict(prefill=0, draft=0, verification=0, decode=0)
    phase = "prefill"
    original_draft = _AdaptiveDraftCache.draft_logits
    original_verify = _AdaptiveDraftCache.verification_logits

    def draft(*args, **kwargs):
        nonlocal phase
        phase = "draft"
        try:
            result = original_draft(*args, **kwargs)
            observed_exits.append(result[1])
            return result
        finally:
            phase = "prefill"

    def verify(*args, **kwargs):
        nonlocal phase
        phase = "verification"
        try:
            return original_verify(*args, **kwargs)
        finally:
            phase = "prefill"

    monkeypatch.setattr(_AdaptiveDraftCache, "draft_logits", draft)
    monkeypatch.setattr(_AdaptiveDraftCache, "verification_logits", verify)

    def layer_call(module, args):
        # AR uses depth 2 here; all calls after the first forward are decode.
        current_phase = "decode" if kind == "autoregressive" and observed["total_layer_executions"] >= 2 else phase
        observed_phases[current_phase] += 1
        observed["total_layer_executions"] += 1

    def classifier_call(module, args):
        observed["total_classifier_calls"] += 1

    hooks = [layer.register_forward_pre_hook(layer_call) for layer in model.model.layers]
    if kind == "adaptive":
        hooks.append(strategy.classifier.head.register_forward_pre_hook(classifier_call))
    try:
        results = []
        for _ in range(2):
            observed.update({key: 0 for key in observed})
            observed_phases.update({key: 0 for key in observed_phases})
            observed_exits.clear()
            result = strategy.generate_token_ids(model, prompt, [], config)
            results.append(result)
            assert result.num_tokens_generated == steps
            assert {key: getattr(result, key) for key in observed} == observed
            assert result.layer_executions_by_phase == observed_phases
            assert sum(result.layer_executions_by_phase.values()) == result.total_layer_executions
            assert result.layers_per_token == (observed["total_layer_executions"] / steps if steps else None)
            if kind == "autoregressive":
                assert result.draft_exit_counts == []
            else:
                assert result.draft_exit_counts == [observed_exits.count(depth) for depth in range(1, 5)]
            if kind != "adaptive" or drafts == 0 or steps < 2:
                assert result.total_classifier_calls == 0
            else:
                assert result.total_classifier_calls > 0
        assert results[0] == results[1]
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("kind", ["autoregressive", "fixed", "adaptive"])
@torch.inference_mode()
def test_eos_keeps_work_counts_without_counting_eos_as_output(kind):
    model = tiny_model(layers=4)
    strategy = strategy_for(kind)
    eos = model(torch.tensor([[1, 5, 9]])).logits[0, -1].argmax().item()
    result = strategy.generate_token_ids(
        model, [1, 5, 9], [eos],
        GenerationConfig(max_steps=8, num_speculations=3, exit_layer=4, sample=False),
    )
    assert result.predicted_tokens == []
    assert result.num_tokens_generated == 0
    assert result.total_layer_executions > 0
    assert result.layers_per_token is None
    response = GenerationResult(result, "", 0, 1.0, None, 0.0)
    serialized = asdict(response)
    assert serialized["total_layer_executions"] == result.total_layer_executions
    assert serialized["total_classifier_calls"] == result.total_classifier_calls
    assert serialized["layers_per_token"] is None
    assert serialized["layer_executions_by_phase"] == result.layer_executions_by_phase
    assert sum(result.layer_executions_by_phase.values()) == result.total_layer_executions
    if kind != "autoregressive":
        assert serialized["draft_exit_statistics"]["total_drafts"] > 0


@torch.inference_mode()
def test_full_depth_acceptance_counts_prefill_and_decode_calls():
    model = tiny_model(layers=4)
    result = SelfSpeculativeGenerationStrategy("off").generate_token_ids(
        model, [1, 5, 9], [],
        GenerationConfig(max_steps=8, num_speculations=3, exit_layer=4, sample=False),
    )
    assert result.acceptance_rate == 1
    assert result.total_layer_executions == (1 + 8) * 4
    assert result.total_classifier_calls == 0
    assert result.draft_exit_counts == [0, 0, 0, 6]
    assert result.layer_executions_by_phase == dict(prefill=4, draft=24, verification=8, decode=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("kind,mode,threshold", [
    ("fixed", "auto", 0), ("adaptive", "auto", 0), ("adaptive", "auto", 1),
    ("adaptive", "on", 0), ("adaptive", "on", 1),
])
@torch.inference_mode()
def test_graph_counts_match_eager_and_exclude_capture_and_previous_requests(kind, mode, threshold):
    model = tiny_model(device="cuda", dtype=torch.float16, layers=4)
    if mode == "on" and not conditional_graphs_available(model):
        pytest.skip("Requires conditional graph API")
    strategy = strategy_for(kind, threshold=threshold)
    config = GenerationConfig(max_steps=8, num_speculations=2, exit_layer=4, sample=False)
    expected = strategy.generate_token_ids(model, [1, 5, 9], [], config)
    strategy.gpu_branching = mode
    strategy.prepare_for_generation(model, [1, 5, 9], config)
    for _ in range(2):
        actual = strategy.generate_token_ids(model, [1, 5, 9], [], config)
        assert actual == expected
