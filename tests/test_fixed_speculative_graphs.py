"""Fixed-depth decoding uses shared verification and graphs without a classifier."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LogitsProcessorList, NoRepeatNGramLogitsProcessor

from self_speculation.early_exit_cuda_graphs import GroupedDraftGraphs
from self_speculation.generator_base import GenerationConfig, GenerationStrategyResult, HuggingfaceLlamaGenerator
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from tests.test_early_exit_latency import tiny_model


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"))])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("depth", [2, 4])
@pytest.mark.parametrize("sample", [False, True])
def test_fixed_depth_matches_target_with_rejection_and_processors(device, attention, depth, sample, monkeypatch):
    # A fixed draft must never load an early-exit checkpoint.
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Unexpected classifier load"))
    model = tiny_model(attention, device, torch.float16 if device == "cuda" else torch.float32, layers=4)
    ids = torch.tensor([[1, 5, 9]], device=device)
    processors = LogitsProcessorList([NoRepeatNGramLogitsProcessor(2)])
    expected = model.generate(ids, max_new_tokens=12, do_sample=False, pad_token_id=0, logits_processor=processors)[0, 3:].tolist()
    config = GenerationConfig(max_steps=12, exit_layer=depth, num_speculations=3, sample=sample, top_k=1)
    for mode in ("off", "auto"):
        strategy = SelfSpeculativeGenerationStrategy(cuda_graphs=mode)
        result = strategy.generate_token_ids(model, ids[0].tolist(), [], config, logits_processors=processors)
        assert result.predicted_tokens == expected
        assert strategy.classifier is None
        if depth < 4:
            assert result.acceptance_rate < 1
        else:
            assert result.acceptance_rate == 1
        if device == "cuda" and mode == "auto":
            assert isinstance(strategy._cuda_graphs, GroupedDraftGraphs)
            assert strategy._cuda_graphs.checks == [depth]
            assert strategy._cuda_graphs.gru_state is None
        else:
            assert strategy._cuda_graphs is None


@pytest.mark.parametrize("steps,drafts", [(0, 3), (1, 3), (8, 0)])
def test_fixed_depth_zero_drafts_and_short_budgets(steps, drafts):
    model = tiny_model(layers=4)
    result = SelfSpeculativeGenerationStrategy().generate_token_ids(
        model, [1, 5, 9], [], GenerationConfig(max_steps=steps, exit_layer=2, num_speculations=drafts, sample=False),
    )
    assert len(result.predicted_tokens) == steps
    assert result.acceptance_rate == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@torch.inference_mode()
def test_fixed_graph_replay_has_no_classifier_or_host_exit_checks():
    model = tiny_model(device="cuda", dtype=torch.float16)
    strategy = SelfSpeculativeGenerationStrategy()
    config = GenerationConfig(max_steps=8, exit_layer=4, num_speculations=3, sample=False)
    strategy.prepare_for_generation(model, [1], config)
    engine = strategy._cuda_graphs
    assert len({graph.pool() for graph in [*engine.draft_graphs.values(), *engine.verify_graphs.values(), *engine.head_graphs.values()]}) == 1
    engine.begin_round(torch.tensor([[1]], device="cuda"))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        _, depth = engine.draft_logits(None, .5, 4)
        torch.cuda.synchronize()
    assert depth == 4
    assert not any(event.key in ("aten::item", "aten::_local_scalar_dense") for event in profile.key_averages())
    assert not any("gru" in event.key.lower() for event in profile.key_averages())


def test_fixed_graph_cli_option():
    from generate import process_cli_arguments
    _, options, config = process_cli_arguments([
        "--model", "unused", "--generation_strategy", "self_speculative", "--exit_layer", "8",
        "--num_speculations", "6", "--ss_cuda_graphs", "off",
    ])
    assert options.ss_cuda_graphs == "off" and config.exit_layer == 8
    with pytest.raises(ValueError, match="ss_cuda_graphs"):
        SelfSpeculativeGenerationStrategy(cuda_graphs="on")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_generator_times_completed_work_after_setup(monkeypatch, device):
    events = []
    class Strategy:
        def prepare_for_generation(self, *args):
            events.append("prepare")
        def generate_token_ids(self, **kwargs):
            events.append("generate")
            return GenerationStrategyResult([3, 4], .5)
    class Tokenizer:
        eos_token_id = 2
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1]])}
        def decode(self, tokens):
            return "answer"
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device(device))]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))
    def timer():
        events.append("timer")
        return 10.0 if events.count("timer") == 1 else 12.0
    monkeypatch.setattr("self_speculation.generator_base.time.perf_counter", timer)
    result = HuggingfaceLlamaGenerator(Tokenizer(), model, Strategy()).generate("prompt", GenerationConfig())
    expected = ["prepare", "synchronize", "timer", "generate", "synchronize", "timer"] if device == "cuda" else ["prepare", "timer", "generate", "timer"]
    assert events == expected
    assert result.total_time == 2 and result.tokens_per_second == 1


def test_comparison_records_each_repeat_and_separates_setup():
    from compare_speculative import measure_strategy, parse_args
    model = tiny_model(layers=4)
    row = measure_strategy(model, SelfSpeculativeGenerationStrategy("off"), [1, 5, 9], [],
                           GenerationConfig(max_steps=8, exit_layer=2, num_speculations=3, sample=False), 1, 2, 42)
    assert len(row["samples"]) == 2
    assert row["samples"][0]["token_ids"] == row["samples"][1]["token_ids"]
    assert row["setup_seconds"] >= 0 and row["median_seconds"] > 0
    assert row["reserved_mib"] is None
    with pytest.raises(SystemExit):
        parse_args(["--repeats", "0"])


def test_comparison_main_writes_complete_report(tmp_path, monkeypatch):
    import json
    import transformers
    import compare_speculative
    from train_ee_cnn import EarlyExitGRU
    model = tiny_model(layers=4)
    classifier = EarlyExitGRU(16, 4)
    checkpoint = tmp_path / "exit.pt"
    torch.save({"model_config": classifier.config, "model_state_dict": classifier.state_dict()}, checkpoint)
    class Tokenizer:
        eos_token_id = 0
        def __call__(self, *args, **kwargs):
            return transformers.BatchEncoding({"input_ids": torch.tensor([[1, 5, 9]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)})
    monkeypatch.setattr(compare_speculative.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(compare_speculative.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    output = tmp_path / "report.json"
    args = compare_speculative.parse_args([
        "--device", "cpu", "--exit-layer", "2", "--max-new-tokens", "8",
        "--repeats", "2", "--warmup", "0", "--ee-checkpoint", str(checkpoint), "--output", str(output),
    ])
    report = compare_speculative.main(args)
    assert json.loads(output.read_text()) == report
    assert len(report["results"]) == 2
    assert {row["config"]["generation_strategy"] for row in report["results"]} == {"self_speculative", "early_exit_speculative"}
    assert all(row["greedy_matches_target"] and len(row["samples"]) == 2 for row in report["results"])
