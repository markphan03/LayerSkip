"""CUDA correctness and memory regressions for the default adaptive backend."""

import gc

import pytest
import torch
from transformers.cache_utils import DynamicCache

from self_speculation.early_exit_cuda_graphs import GroupedDraftGraphs
from self_speculation.early_exit_speculation_generator import (
    EarlyExitSelfSpeculativeGenerationStrategy, _AdaptiveDraftCache, exit_check_layers,
)
from self_speculation.generator_base import GenerationConfig
from tests.test_early_exit_latency import tiny_model
from train_ee_cnn import EarlyExitGRU


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("max_depth", [4, 8])
@torch.inference_mode()
def test_grouped_graphs_match_adaptive_cache_across_exits_and_rollback(attention, max_depth):
    model = tiny_model(attention, "cuda", torch.float16)
    classifier = EarlyExitGRU(16, 4, num_layers=2).cuda().half().eval()
    classifier.head.weight.zero_()
    engine = GroupedDraftGraphs(model, classifier, exit_check_layers(8, max_depth), .5, 32, 3)
    ids = torch.tensor([[1, 5, 9]], device="cuda")
    engine.prefill(ids)
    cache = DynamicCache()
    # Alternate shallow/deep exits, then overwrite a rejected suffix. This
    # exercises shared-pool graphs in orders different from capture order.
    for _ in range(2):
        engine.begin_round(ids)
        eager = _AdaptiveDraftCache(model, cache, ids[:, cache.get_seq_length():])
        full = ids.clone()
        for index, (bias, token) in enumerate([(10, 3), (-10, 7), (10, 2)]):
            classifier.head.bias.fill_(bias)
            expected, depth = eager.draft_logits(classifier, .5, max_depth)
            actual, actual_depth = engine.draft_logits(classifier, .5, max_depth)
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            assert actual_depth == depth == (2 if bias > 0 else max_depth)
            if index == 0:
                assert engine.completed == {start: int(start == 0) for start, _ in engine.groups}
            eager.append(token)
            engine.append(token)
            full = torch.cat([full, full.new_tensor([[token]])], dim=1)
        actual = engine.verification_logits(4)
        expected = eager.verification_logits(4)
        baseline = model(full, use_cache=True)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(actual, baseline.logits[:, -4:], atol=2e-3, rtol=2e-3)
        for layer in range(8):
            for actual_kv, expected_kv in zip(
                (engine.cache.key_cache[layer], engine.cache.value_cache[layer]),
                baseline.past_key_values[layer],
            ):
                torch.testing.assert_close(actual_kv[:, :, :full.shape[1]], expected_kv, atol=2e-3, rtol=2e-3)
        ids = torch.cat([ids, ids.new_tensor([[3, 11]])], dim=1)
        cache.crop(ids.shape[1] - 1)


@pytest.mark.parametrize("drafts", [0, 3])
@torch.inference_mode()
def test_auto_works_without_conditional_api_and_matches_target(tmp_path, monkeypatch, drafts):
    monkeypatch.setattr("self_speculation.early_exit_cuda_graphs.conditional_graphs_available", lambda model: False)
    model = tiny_model(device="cuda", dtype=torch.float16)
    classifier = EarlyExitGRU(16, 4)
    path = tmp_path / "classifier.pt"
    torch.save({"model_config": classifier.config, "model_state_dict": classifier.state_dict()}, path)
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold=0, gpu_branching="auto")
    prompt = [1, 5, 9]
    expected = model.generate(
        torch.tensor([prompt], device="cuda"), max_new_tokens=12, do_sample=False, pad_token_id=0,
    )[0, len(prompt):].tolist()
    for sample in [False, True]:
        result = strategy.generate_token_ids(
            model, prompt, [], GenerationConfig(max_steps=12, num_speculations=drafts, sample=sample, top_k=1),
        )
        assert result.predicted_tokens == expected
        assert isinstance(strategy._cuda_graphs, GroupedDraftGraphs)


@torch.inference_mode()
def test_grouped_graphs_release_pool_after_repeated_rebuilds():
    model = tiny_model(device="cuda", dtype=torch.float16)
    classifier = EarlyExitGRU(16, 4).cuda().half().eval()
    # Warm CUDA libraries before recording the baseline. Stream/BLAS caches
    # may persist, so allow a small fixed overhead, not gigabytes per rebuild.
    model(torch.tensor([[1, 2]], device="cuda"))
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_reserved()
    for capacity, drafts in [(32, 2), (64, 6), (32, 3)]:
        engine = GroupedDraftGraphs(model, classifier, exit_check_layers(8), .5, capacity, drafts)
        graphs = [*engine.draft_graphs.values(), *engine.verify_graphs.values(), *engine.head_graphs.values()]
        assert len({graph.pool() for graph in graphs}) == 1
        pool = engine.pool
        del graphs
        engine.begin_round(torch.tensor([[1]], device="cuda"))
        engine.draft_logits(classifier, .5, 8)
        torch.cuda.synchronize()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        assert not any(tuple(segment["segment_pool_id"]) == pool for segment in torch.cuda.memory_snapshot())
        assert torch.cuda.memory_reserved() - baseline < 64 * 1024**2
