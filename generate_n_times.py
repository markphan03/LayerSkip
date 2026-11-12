"""Generate N independent completions of one prompt with a single model load."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import statistics
from typing import Optional

import torch
import transformers

from arguments import Arguments, simple_parse_args_string
from generate import GenerateArguments, StreamerType, load_model_and_tokenizer, setup
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.early_exit_speculation_generator import EarlyExitSelfSpeculativeGenerationStrategy
from self_speculation.generator_base import GenerationConfig, HuggingfaceLlamaGenerator
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.speculative_streamer import SpeculativeTextStreamer


@dataclass
class RepeatArguments:
    prompt: str = field(metadata={"help": "Prompt reused independently for every completion."})
    N: int = field(default=1, metadata={"help": "Number of reported completions."})
    warmup: int = field(default=0, metadata={"help": "Unreported warmup generations before the N runs."})
    output: Optional[str] = field(default=None, metadata={"help": "Optional JSON report path."})


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
        if not 0 < generate_args.ee_initial_fraction <= 1:
            parser.error("--ee_initial_fraction must be greater than 0 and at most 1.")
        if config.exit_layer != -1 and config.exit_layer < 1:
            parser.error("--exit_layer must be -1 or a positive maximum draft depth.")
    if config.sample and config.temperature <= 0:
        parser.error("--temperature must be positive when sampling.")
    if config.top_k < 0 or not 0 <= config.top_p <= 1:
        parser.error("--top_k must be non-negative and --top_p must be between 0 and 1.")
    args.model_args = simple_parse_args_string(args.model_args) if args.model_args else {}
    return args, generate_args, config, repeat


def main(args, generate_args, config, repeat):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    if config.generation_strategy == "early_exit_speculative":
        strategy = EarlyExitSelfSpeculativeGenerationStrategy(
            checkpoint_path=generate_args.ee_checkpoint,
            threshold=generate_args.ee_threshold,
            initial_fraction=generate_args.ee_initial_fraction,
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
        acceptance = response.generation_strategy_result.acceptance_rate
        if acceptance is not None:
            print(f"Acceptance rate: {acceptance:.2%}", flush=True)

    total_tokens = sum(row["num_tokens_generated"] for row in runs)
    total_time = sum(row["total_time"] for row in runs)
    acceptance_rates = [
        row["generation_strategy_result"]["acceptance_rate"] for row in runs
        if row["generation_strategy_result"]["acceptance_rate"] is not None
    ]
    summary = {
        "num_runs": len(runs),
        "total_tokens": total_tokens,
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
    print("Timing excludes model loading, tokenization, decoding, graph setup, and warmup runs.")
    report = {
        "arguments": asdict(args), "generate_arguments": asdict(generate_args),
        "generation_config": asdict(config), "repeat_arguments": asdict(repeat),
        "summary": summary, "runs": runs,
    }
    if repeat.output:
        path = Path(repeat.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved report to {path}")
    return report


if __name__ == "__main__":
    main(*process_cli_arguments())
