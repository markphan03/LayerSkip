"""Offline checks for adaptive caches, verification, sampling, and the CLI."""

from io import StringIO

import pytest
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

import generate
from self_speculation.early_exit_speculation_generator import (
    EarlyExitSelfSpeculativeGenerationStrategy,
    _AdaptiveDraftCache,
    _verify_sample,
)
from self_speculation.generator_base import GenerationConfig
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from train_ee_cnn import EarlyExitGRU


@pytest.fixture(params=["eager", "sdpa"])
def components(request, tmp_path):
    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, eos_token_id=None,
    )
    config._attn_implementation = request.param
    model = LlamaForCausalLM(config).eval()
    classifier = EarlyExitGRU(16, 4, num_layers=2).eval()
    path = tmp_path / "exit.pt"
    torch.save({
        "model_config": classifier.config,
        "model_state_dict": classifier.state_dict(),
    }, path)
    return model, classifier, path


@torch.inference_mode()
def test_changing_depths_backfill_and_rollback_match_full_model(components):
    model, classifier, _ = components
    ids = torch.tensor([[1, 5, 9, 3, 7, 2, 4, 6]])
    cache = DynamicCache()
    counts = [0] * len(model.model.layers)
    hooks = []
    for index, layer in enumerate(model.model.layers):
        def count(module, args, index=index):
            counts[index] += args[0].shape[1]
        hooks.append(layer.register_forward_pre_hook(count))
    try:
        # Round two starts after discarding the last two verified positions.
        for start, end in [(0, 6), (4, 8)]:
            counts[:] = [0] * len(counts)
            runner = _AdaptiveDraftCache(model, cache, ids[:, start:end - 3])
            observed = []
            for position, depth in zip(range(end - 3, end), [1, 3, 2]):
                logits, actual_depth = runner.draft_logits(classifier, 1.0, depth)
                assert actual_depth == depth
                observed.append((position, depth, logits))
                runner.append(ids[0, position].item())
            verified = runner.verification_logits(4)
            # Each layer computes each new position once despite changing depth.
            assert counts == [end - start] * len(counts)

            # A separate full pass supplies raw outputs at every decoder layer.
            raw = []
            captures = [
                layer.register_forward_hook(lambda module, args, output: raw.append(output[0]))
                for layer in model.model.layers
            ]
            try:
                expected = model(ids[:, :end], use_cache=True)
            finally:
                for hook in captures:
                    hook.remove()
            torch.testing.assert_close(verified, expected.logits[:, end - 4:])
            for position, depth, logits in observed:
                reference = model.lm_head(model.model.norm(raw[depth - 1][:, position - 1]))
                torch.testing.assert_close(logits, reference)
            for index in range(len(model.model.layers)):
                for actual, reference in zip(cache[index], expected.past_key_values[index]):
                    torch.testing.assert_close(actual, reference)
            cache.crop(end - 2)
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def test_gru_decisions_match_training_forward_and_reset_each_token(components):
    model, classifier, _ = components
    ids = torch.tensor([[1, 5, 9]])
    runner = _AdaptiveDraftCache(model, DynamicCache(), ids)
    for token in [3, 7, 2]:
        raw = []
        hooks = [
            layer.register_forward_hook(lambda module, args, output: raw.append(output[0][:, -1:]))
            for layer in model.model.layers
        ]
        try:
            model(ids, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        scores = classifier(torch.stack(raw, dim=1))[0, :, 0].sigmoid()
        threshold = scores[:-1].mean().item()
        eligible = scores.ge(threshold)
        eligible[-1] = True
        depth = eligible.long().argmax().item() + 1
        logits, actual_depth = runner.draft_logits(classifier, threshold, len(raw))
        assert actual_depth == depth
        torch.testing.assert_close(logits, model.lm_head(model.model.norm(raw[depth - 1]))[:, 0])
        runner.append(token)
        ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)


@pytest.mark.parametrize("threshold,budget", [(0.0, 3), (0.5, 6), (1.0, 3), (0.0, 0)])
@torch.inference_mode()
def test_greedy_matches_full_model_with_processors_and_budget(components, threshold, budget):
    model, _, path = components
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold)
    config = GenerationConfig(max_steps=13, sample=False, num_speculations=budget)
    processors = transformers.LogitsProcessorList([
        transformers.NoRepeatNGramLogitsProcessor(2),
    ])
    expected = model.generate(
        torch.tensor([[1, 5, 9]]), max_new_tokens=13, do_sample=False,
        logits_processor=processors, pad_token_id=0,
    )[0, 3:].tolist()
    result = strategy.generate_token_ids(model, [1, 5, 9], [], config, processors)
    assert result.predicted_tokens == expected
    assert 0 <= result.acceptance_rate <= 1
    if threshold == 1 and budget:
        assert result.acceptance_rate == 1
    if budget == 0:
        assert result.acceptance_rate == 0


@pytest.mark.parametrize("steps", [0, 1])
def test_short_budgets_have_no_drafts_or_division_by_zero(components, steps):
    model, _, path = components
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path)
    result = strategy.generate_token_ids(
        model, [1, 5, 9], [], GenerationConfig(max_steps=steps, sample=False, num_speculations=6),
    )
    assert len(result.predicted_tokens) == steps
    assert result.acceptance_rate == 0


@torch.inference_mode()
def test_portable_prefill_keeps_prompt_out_of_adaptive_state_storage(components, monkeypatch):
    model, _, path = components
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold=1, gpu_branching="off")
    prompt = [1, 5, 9] * 8
    lengths = []
    original = _AdaptiveDraftCache.__init__

    def record(self, model, cache, input_ids):
        lengths.append(input_ids.shape[1])
        original(self, model, cache, input_ids)

    monkeypatch.setattr(_AdaptiveDraftCache, "__init__", record)
    expected = model.generate(
        torch.tensor([prompt]), max_new_tokens=12, do_sample=False, pad_token_id=0,
    )[0, len(prompt):].tolist()
    result = strategy.generate_token_ids(
        model, prompt, [], GenerationConfig(max_steps=12, num_speculations=3, sample=False),
    )
    assert result.predicted_tokens == expected
    assert lengths and set(lengths) == {1}


def test_sample_top_k_one_matches_greedy_even_when_drafts_are_rejected(components):
    model, _, path = components
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold=0)
    greedy = strategy.generate_token_ids(
        model, [1, 5, 9], [], GenerationConfig(max_steps=16, sample=False, num_speculations=3),
    )
    sampled = strategy.generate_token_ids(
        model, [1, 5, 9], [], GenerationConfig(max_steps=16, sample=True, top_k=1, num_speculations=3),
    )
    assert sampled.predicted_tokens == greedy.predicted_tokens
    assert sampled.acceptance_rate < 1  # Exercise actual rejection and rollback.


def test_sampling_acceptance_and_residual_distribution(monkeypatch):
    target = torch.tensor([[0.1, 0.3, 0.6]])
    draft = torch.tensor([[0.6, 0.3, 0.1]])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.9))
    seen = []

    def sample(probabilities, num_samples):
        seen.append(probabilities)
        return torch.tensor([[2]])

    monkeypatch.setattr(torch, "multinomial", sample)
    assert _verify_sample(0, target, draft) == (2, False)
    torch.testing.assert_close(seen[0], torch.tensor([[0.0, 0.0, 1.0]]))
    assert _verify_sample(1, target, draft) == (1, True)
    assert _verify_sample(2, target, draft) == (2, True)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.1))
    assert _verify_sample(0, target, draft) == (0, True)


def test_sampling_marginal_matches_target_distribution():
    torch.manual_seed(17)
    target = torch.tensor([[0.1, 0.3, 0.6]])
    draft = torch.tensor([[0.6, 0.3, 0.1]])
    counts = torch.zeros(3)
    for token in torch.multinomial(draft[0], num_samples=6000, replacement=True).tolist():
        output, _ = _verify_sample(token, target, draft)
        counts[output] += 1
    torch.testing.assert_close(counts / counts.sum(), target[0], atol=0.025, rtol=0)


@pytest.mark.parametrize("speculative", [False, True])
def test_stopping_and_streaming_only_commit_verified_tokens(components, speculative):
    model, _, path = components
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold=0)
    received, drafts, deletions = [], [], []

    class Recorder:
        def put(self, tokens, is_draft=False):
            (drafts if is_draft else received).extend(tokens.flatten().tolist())

        def delete(self, count):
            deletions.append(count)

    class DraftRecorder(Recorder, SpeculativeTextStreamer):
        def __init__(self):
            pass

    streamer = DraftRecorder() if speculative else Recorder()

    def stop(ids, scores):
        assert ids[0, :3].tolist() == [1, 5, 9]
        return torch.tensor([ids.shape[1] >= 5])

    result = strategy.generate_token_ids(
        model, [1, 5, 9], [], GenerationConfig(max_steps=10, sample=False, num_speculations=3),
        stopping_criteria=stop, streamer=streamer,
    )
    assert len(result.predicted_tokens) == 2
    assert received == result.predicted_tokens
    if speculative:
        assert len(drafts) == sum(deletions) > 0


@pytest.mark.parametrize("threshold", [0, 1])
def test_eos_from_target_stops_without_emitting_eos(components, threshold):
    model, _, path = components
    strategy = EarlyExitSelfSpeculativeGenerationStrategy(path, threshold)
    with torch.inference_mode():
        eos = model(torch.tensor([[1, 5, 9]])).logits[0, -1].argmax().item()
    model.generation_config.eos_token_id = [eos]
    result = strategy.generate_token_ids(
        model, [1, 5, 9], [], GenerationConfig(max_steps=10, sample=False, num_speculations=3),
    )
    assert result.predicted_tokens == []


def test_cli_reads_real_arguments_and_preserves_fixed_layer_option(monkeypatch):
    for name in ("self_speculative", "early_exit_speculative"):
        monkeypatch.setattr("sys.argv", [
            "generate.py", "--model", "facebook/layerskip-llama2-7B", "--sample", "True",
            "--max_steps", "512", "--generation_strategy", name,
            "--exit_layer", "8", "--num_speculations", "6",
            "--ee_checkpoint", "some/checkpoint.pt", "--ee_threshold", "0.7",
        ])
        args, options, config = generate.process_cli_arguments()
        assert args.model == "facebook/layerskip-llama2-7B"
        assert config.sample and config.max_steps == 512
        assert config.generation_strategy == name
        assert config.exit_layer == 8 and config.num_speculations == 6
        assert options.ee_checkpoint == "some/checkpoint.pt" and options.ee_threshold == 0.7


def test_cli_invalid_threshold_fails_before_loading_models():
    with pytest.raises(SystemExit):
        generate.process_cli_arguments([
            "--model", "unused", "--generation_strategy", "early_exit_speculative",
            "--num_speculations", "6", "--ee_threshold", "2",
        ])


def test_main_routes_new_strategy_and_prints_acceptance_for_empty_answer(components, monkeypatch, capsys):
    model, _, path = components

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, **kwargs):
            return transformers.BatchEncoding({"input_ids": torch.tensor([[1, 5, 9]])})

    captured = []

    class Generator:
        def __init__(self, tokenizer, model, generation_strategy):
            assert isinstance(generation_strategy, EarlyExitSelfSpeculativeGenerationStrategy)
            captured.append(generation_strategy.threshold)

        def generate(self, **kwargs):
            from self_speculation.generator_base import GenerationResult, GenerationStrategyResult
            return GenerationResult(GenerationStrategyResult([], 0.0), "", 0, 1.0, None, 0.0)

    monkeypatch.setattr(generate, "setup", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate, "load_model_and_tokenizer", lambda *args, **kwargs: (model, Tokenizer()))
    monkeypatch.setattr(generate, "HuggingfaceLlamaGenerator", Generator)
    monkeypatch.setattr(model, "generate", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate.sys, "stdin", StringIO("Question: Why?\nAnswer:"))
    args, options, config = generate.process_cli_arguments([
        "--model", "local", "--generation_strategy", "early_exit_speculative",
        "--num_speculations", "3", "--ee_checkpoint", str(path), "--ee_threshold", "0.7",
        "--streamer", "none",
    ])
    generate.main(args, options, config)
    assert captured == [0.7]
    assert "Acceptance Rate: 0.00%" in capsys.readouterr().out
