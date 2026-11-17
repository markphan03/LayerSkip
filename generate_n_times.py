"""Generate N independent completions of one prompt with a single model load."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Optional

import torch
import transformers

from arguments import Arguments, simple_parse_args_string
from generate import GenerateArguments, StreamerType, load_model_and_tokenizer, setup
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.early_exit_speculation_generator import EarlyExitSelfSpeculativeGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig, HuggingfaceLlamaGenerator,
    empty_layer_execution_counts, format_layer_execution_counts,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.exit_statistics import format_draft_exit_statistics, summarize_draft_exits


@dataclass
class RepeatArguments:
    prompt: str = field(metadata={"help": "Prompt reused independently for every completion."})
    N: int = field(default=1, metadata={"help": "Number of reported completions."})
    warmup: int = field(default=0, metadata={"help": "Unreported warmup generations before the N runs."})
    output: Optional[str] = field(default=None, metadata={
        "help": "Optional output path: .txt saves terminal output; other extensions save a JSON report."
    })


@dataclass
class RepeatGenerateArguments(GenerateArguments):
    streamer: StreamerType = StreamerType.NONE


def process_cli_arguments(argv=None):
    parser = transformers.HfArgumentParser(
        (Arguments, RepeatGenerateArguments, GenerationConfig, RepeatArguments),
        description=__doc__,
    )
    args, generate_args, config, repeat = parser.parse_args_into_dataclasses(args=argv)
    if repeat.N < 1 or repeat.warmup < 0:
        parser.error("--N must be positive and --warmup must be non-negative.")
    if not repeat.prompt.strip():
        parser.error("--prompt must be nonempty.")
    if config.max_steps < 0:
        parser.error("--max_steps must be non-negative.")
    if config.generation_strategy not in ("autoregressive", "self_speculative", "early_exit_speculative"):
        parser.error("--generation_strategy must be autoregressive, self_speculative, or early_exit_speculative.")
    if config.generation_strategy != "autoregressive" and config.num_speculations < 0:
        parser.error("Specify a non-negative --num_speculations for speculative decoding.")
    if config.generation_strategy == "self_speculative" and config.exit_layer < 1:
        parser.error("Fixed-depth self_speculative requires a positive --exit_layer.")
    if config.generation_strategy == "early_exit_speculative":
        if not 0 <= generate_args.ee_threshold <= 1:
            parser.error("--ee_threshold must be between 0 and 1.")
        if generate_args.ee_classifier_start_layer < 1:
            parser.error("--ee-classifier-start-layer must be a positive integer (1-based).")
        if config.exit_layer != -1 and config.exit_layer < 1:
            parser.error("--exit_layer must be -1 or a positive maximum draft depth.")
    if config.sample and config.temperature <= 0:
        parser.error("--temperature must be positive when sampling.")
    if config.top_k < 0 or not 0 <= config.top_p <= 1:
        parser.error("--top_k must be non-negative and --top_p must be between 0 and 1.")
    args.model_args = simple_parse_args_string(args.model_args) if args.model_args else {}
    return args, generate_args, config, repeat


class _TeeStream:
    """Display output normally and flush a copy to the log as it arrives."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def main(args, generate_args, config, repeat):
    if repeat.output and Path(repeat.output).suffix.lower() == ".txt":
        path = Path(repeat.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as log:
            with redirect_stdout(_TeeStream(sys.stdout, log)), redirect_stderr(_TeeStream(sys.stderr, log)):
                report = _run_generations(args, generate_args, config, repeat)
                print(f"Saved terminal output to {path}", flush=True)
                return report
    return _run_generations(args, generate_args, config, repeat)


def _run_generations(args, generate_args, config, repeat):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    if config.generation_strategy == "early_exit_speculative":
        strategy = EarlyExitSelfSpeculativeGenerationStrategy(
            checkpoint_path=generate_args.ee_checkpoint,
            threshold=generate_args.ee_threshold,
            classifier_start_layer=generate_args.ee_classifier_start_layer,
            gpu_branching=generate_args.ee_gpu_branching,
        )
        strategy.prepare_model(model)
    elif config.generation_strategy == "self_speculative":
        if config.exit_layer > len(model.model.layers):
            raise ValueError("--exit_layer exceeds the model's number of layers.")
        strategy = SelfSpeculativeGenerationStrategy(cuda_graphs=generate_args.ss_cuda_graphs)
    else:
        strategy = AutoRegressiveGenerationStrategy()
    generator = HuggingfaceLlamaGenerator(tokenizer, model, strategy)

    print(f"Strategy: {config.generation_strategy}; completions: {repeat.N}", flush=True)
    for index in range(repeat.warmup):
        print(f"Warmup {index + 1}/{repeat.warmup}", flush=True)
        generator.generate(prompt=repeat.prompt, generation_config=config)

    runs = []
    for index in range(repeat.N):
        # Distinct, reproducible random streams, independent of warmup count.
        seed = args.seed + index
        random.seed(seed)
        torch.manual_seed(seed)
        show_result = index + 1 < 5
        streamer = None
        if show_result and generate_args.streamer == StreamerType.STANDARD:
            streamer = transformers.TextStreamer(tokenizer)
        elif show_result and generate_args.streamer == StreamerType.SPECULATIVE:
            streamer = SpeculativeTextStreamer(tokenizer)
        print(f"\n=== Run {index + 1}/{repeat.N} (seed {seed}) ===", flush=True)
        response = generator.generate(
            prompt=repeat.prompt, generation_config=config, streamer=streamer,
        )
        if streamer is not None:
            streamer.end()
        elif show_result:
            print(response.decoded_prediction)
        row = {"run": index + 1, "seed": seed, **asdict(response)}
        runs.append(row)
        time_per_token = (
            f"{response.time_per_token:.6f}s/token"
            if response.time_per_token is not None else "N/A"
        )
        print(f"Time: {response.total_time:.3f}s | Tokens: {response.num_tokens_generated} | "
              f"Time per token: {time_per_token} | Tokens/s: {response.tokens_per_second:.3f}",
              flush=True)
        layers_per_token = f"{response.layers_per_token:.3f}" if response.layers_per_token is not None else "N/A"
        print(f"Layer executions: {response.total_layer_executions} | Layers/token: {layers_per_token} | "
              f"Classifier calls: {response.total_classifier_calls}",
              flush=True)
        acceptance = response.generation_strategy_result.acceptance_rate
        print(format_layer_execution_counts(response.layer_executions_by_phase), flush=True)
        if acceptance is not None:
            print(f"Acceptance rate: {acceptance:.2%}", flush=True)
        if response.draft_exit_statistics is not None:
            print(format_draft_exit_statistics(response.draft_exit_statistics), flush=True)

    total_tokens = sum(row["num_tokens_generated"] for row in runs)
    total_time = sum(row["total_time"] for row in runs)
    total_layers = sum(row["total_layer_executions"] for row in runs)
    phase_counts = empty_layer_execution_counts()
    for row in runs:
        for phase, count in row["layer_executions_by_phase"].items():
            phase_counts[phase] += count
    total_classifier_calls = sum(row["total_classifier_calls"] for row in runs)
    exit_counts = [row["generation_strategy_result"]["draft_exit_counts"] for row in runs]
    exit_summary = summarize_draft_exits([sum(counts) for counts in zip(*exit_counts)])
    acceptance_rates = [
        row["generation_strategy_result"]["acceptance_rate"] for row in runs
        if row["generation_strategy_result"]["acceptance_rate"] is not None
    ]
    summary = {
        "num_runs": len(runs),
        "total_tokens": total_tokens,
        "total_layer_executions": total_layers,
        "layer_executions_by_phase": phase_counts,
        "total_classifier_calls": total_classifier_calls,
        "draft_exit_statistics": exit_summary,
        "layers_per_token": total_layers / total_tokens if total_tokens else None,
        "total_generation_seconds": total_time,
        "mean_generation_seconds": total_time / len(runs),
        "time_per_token": total_time / total_tokens if total_tokens else None,
        "tokens_per_second": total_tokens / total_time if total_time else 0.0,
        "mean_acceptance_rate": statistics.mean(acceptance_rates) if acceptance_rates else None,
    }
    time_per_token = (
        f"{summary['time_per_token']:.6f}s/token"
        if summary["time_per_token"] is not None else "N/A"
    )
    print(f"\nCompleted {len(runs)} runs | Total tokens: {total_tokens} | "
          f"Generation time: {total_time:.3f}s | \n Time per token: {time_per_token} | "
          f"Tokens/s: {summary['tokens_per_second']:.3f}")
    if acceptance_rates:
        print(f"Mean per-run acceptance rate: {summary['mean_acceptance_rate']:.2%}")
    layers_per_token = f"{summary['layers_per_token']:.3f}" if total_tokens else "N/A"
    print(f"Total layer executions: {total_layers} | Layers/token: {layers_per_token} | "
          f"Total classifier calls: {total_classifier_calls}")
    print(format_layer_execution_counts(phase_counts))
    if len(runs) > 1 and exit_summary is not None:
        print(format_draft_exit_statistics(exit_summary, title="Draft exit depths across all runs"))
    print("Work counts include prompt prefill and discarded drafts; classifier calls count GRU exit decisions.")
    print("Timing excludes model loading, tokenization, decoding, graph setup, and warmup runs.")
    report = {
        "arguments": asdict(args), "generate_arguments": asdict(generate_args),
        "generation_config": asdict(config), "repeat_arguments": asdict(repeat),
        "summary": summary, "runs": runs,
    }
    if repeat.output and Path(repeat.output).suffix.lower() != ".txt":
        path = Path(repeat.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved report to {path}")
    return report


if __name__ == "__main__":
    main(*process_cli_arguments())
