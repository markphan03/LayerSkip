"""Benchmark fixed and classifier-controlled LayerSkip on reproducible inputs.

An --evaluation_file is created once and reused verbatim on subsequent runs.
The original benchmark.py is intentionally kept unchanged.
"""

# python benchmark2.py \
#   --model facebook/layerskip-llama2-7B \
#   --dataset cnn_dm_summarization \
#   --num_samples 200 \
#   --evaluation_file benchmark_inputs_200.jsonl \
#   --generation_strategy self_speculative \
#   --exit_layer 8 \
#   --num_speculations 6 \
#   --max_steps 512 \
#   --sample true \
#   --seed 42 \
#   --warmup 0 \
#   --output logs/benchmark2_fixed.json

# # Classifier-controlled LayerSkip, after extending benchmark.py
# python benchmark2.py \
#   --model facebook/layerskip-llama2-7B \
#   --dataset cnn_dm_summarization \
#   --num_samples 200 \
#   --evaluation_file benchmark_inputs_200.jsonl \
#   --generation_strategy early_exit_speculative \
#   --ee_checkpoint checkpoints/ee_cnn/best_model/final.pt \
#   --ee_threshold 0.8 \
#   --ee_classifier_start_layer 1 \
#   --exit_layer 8 \
#   --num_speculations 6 \
#   --max_steps 512 \
#   --sample true \
#   --seed 42 \
#   --warmup 0 \
#   --output logs/benchmark2_classifier.json

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Optional

import torch
import transformers
from torchmetrics.text import BLEUScore, EditDistance, ROUGEScore
from tqdm import tqdm

from arguments import Arguments, simple_parse_args_string
from data import EvaluationExample, LowercaseProcessingFunction, get_data, get_valid_dataset_formats
from generate import GenerateArguments, StreamerType, load_model_and_tokenizer, setup
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.early_exit_speculation_generator import EarlyExitSelfSpeculativeGenerationStrategy
from self_speculation.exit_statistics import format_draft_exit_statistics, summarize_draft_exits
from self_speculation.generator_base import (
    GenerationConfig, HuggingfaceLlamaGenerator, empty_layer_execution_counts,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy


@dataclass
class BenchmarkArguments:
    dataset: str = "cnn_dm_summarization"
    data_path: Optional[str] = None
    random_shuffle: bool = True
    num_samples: int = 200
    n_shot: int = 0
    template: Optional[str] = None
    evaluation_file: Optional[str] = field(default=None, metadata={
        "help": "JSONL of selected prompts/references: create if missing, otherwise reuse in saved order.",
    })
    prepare_only: bool = field(default=False, metadata={
        "help": "Prepare --evaluation_file and exit without loading the model.",
    })
    warmup: int = 1
    output: Optional[str] = field(default=None, metadata={"help": "JSON report path (default: output_dir/benchmark2_<timestamp>.json)."})
    print_examples: bool = False


@dataclass
class BenchmarkGenerationArguments(GenerateArguments):
    streamer: StreamerType = StreamerType.NONE


def process_cli_arguments(argv=None):
    parser = transformers.HfArgumentParser(
        (Arguments, BenchmarkArguments, BenchmarkGenerationArguments, GenerationConfig),
        description=__doc__,
    )
    args, bench, generate_args, config = parser.parse_args_into_dataclasses(args=argv)
    if bench.dataset not in get_valid_dataset_formats():
        parser.error(f"Unsupported dataset: {bench.dataset}")
    if bench.num_samples < 1 or bench.n_shot < 0 or bench.warmup < 0:
        parser.error("--num_samples must be positive; --n_shot and --warmup must be non-negative.")
    if bench.prepare_only and not bench.evaluation_file:
        parser.error("--prepare_only requires --evaluation_file.")
    if args.seed is None:
        parser.error("--seed must be an integer.")
    if generate_args.streamer != StreamerType.NONE:
        parser.error("Benchmarks require --streamer none; use --print_examples to display responses.")
    if not bench.prepare_only:
        if config.generation_strategy not in ("autoregressive", "self_speculative", "early_exit_speculative"):
            parser.error("Unsupported --generation_strategy.")
        if config.max_steps < 0:
            parser.error("--max_steps must be non-negative.")
        if config.generation_strategy != "autoregressive" and config.num_speculations < 0:
            parser.error("Speculative decoding requires a non-negative --num_speculations.")
        if config.generation_strategy == "self_speculative" and config.exit_layer < 1:
            parser.error("self_speculative requires a positive --exit_layer.")
        if config.generation_strategy == "early_exit_speculative":
            if not 0 <= generate_args.ee_threshold <= 1 or generate_args.ee_classifier_start_layer < 1:
                parser.error("--ee_threshold must be between 0 and 1; --ee_classifier_start_layer must be positive.")
            if config.exit_layer != -1 and config.exit_layer < 1:
                parser.error("--exit_layer must be -1 or a positive draft-depth cap.")
        if config.sample and config.temperature <= 0:
            parser.error("--temperature must be positive when sampling.")
        if config.top_k < 0 or not 0 <= config.top_p <= 1:
            parser.error("--top_k must be non-negative; --top_p must be between 0 and 1.")
    args.model_args = simple_parse_args_string(args.model_args) if args.model_args else {}
    return args, bench, generate_args, config


def _serialize_examples(examples):
    return "".join(json.dumps({"prompt": e.input, "response": e.output}, ensure_ascii=False) + "\n"
                   for e in examples)


def load_evaluation_set(bench, seed):
    """Use a local RNG so model loading and warmup cannot change selection."""
    path = Path(bench.evaluation_file) if bench.evaluation_file else None
    if path is not None and path.exists():
        examples = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or not isinstance(row.get("prompt"), str)
                    or not row["prompt"].strip() or not isinstance(row.get("response"), str)):
                raise ValueError(f"{path}:{line_number}: expected a nonempty prompt and a string response.")
            examples.append(EvaluationExample(input=row["prompt"], output=row["response"]))
        if len(examples) != bench.num_samples:
            raise ValueError(f"{path} contains {len(examples)} examples; --num_samples is {bench.num_samples}. "
                             "Use a different evaluation file or match --num_samples.")
        print(f"Reusing {len(examples)} saved examples from {path} (dataset selection options ignored).", flush=True)
    else:
        examples = get_data(
            random_shuffle=False, num_samples=None, dataset=bench.dataset,
            data_path=bench.data_path, n_shot=bench.n_shot, seed=seed, template=bench.template,
        )
        if bench.random_shuffle:
            random.Random(seed).shuffle(examples)
        if len(examples) < bench.num_samples:
            raise ValueError(f"Requested {bench.num_samples} examples, but dataset contains {len(examples)}.")
        examples = examples[:bench.num_samples]
        if any(not isinstance(e.input, str) or not e.input.strip() or not isinstance(e.output, str)
               for e in examples):
            raise ValueError("Every selected example must have a nonempty prompt and a string reference.")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                handle.write(_serialize_examples(examples))
            print(f"Saved {len(examples)} examples to {path}.", flush=True)
    digest = hashlib.sha256(_serialize_examples(examples).encode("utf-8")).hexdigest()
    return examples, digest


class TextMetrics:
    """Original quality metrics, with corrected argument order and true exact match."""

    def __init__(self):
        self.rouge = ROUGEScore(
            rouge_keys=("rougeL", "rouge1", "rouge2", "rouge3"),
            normalizer=LowercaseProcessingFunction,
        )
        self.bleu = BLEUScore(n_gram=4)
        self.edit_distance = EditDistance()
        self.count = 0
        self.exact_matches = 0

    def update(self, prediction, reference):
        self.rouge.update([prediction], [reference])
        # BLEU expects one list of references per prediction.
        self.bleu.update([prediction], [[reference]])
        self.edit_distance.update([prediction], [reference])
        self.count += 1
        self.exact_matches += prediction == reference

    def compute(self):
        names = {"rouge-l": "rougeL", "rouge-1": "rouge1", "rouge-2": "rouge2", "rouge-3": "rouge3"}
        if not self.count:
            return dict.fromkeys([*names, "bleu_score", "edit_distance", "exact_match"])
        rouge = self.rouge.compute()
        return {
            **{name: rouge[f"{key}_fmeasure"].item() for name, key in names.items()},
            "bleu_score": self.bleu.compute().item(),
            "edit_distance": self.edit_distance.compute().item(),
            "exact_match": self.exact_matches / self.count,
        }


def build_strategy(model, generate_args, config):
    if config.generation_strategy != "autoregressive" and config.exit_layer > len(model.model.layers):
        raise ValueError("--exit_layer exceeds the model's number of layers.")
    if config.generation_strategy == "early_exit_speculative":
        strategy = EarlyExitSelfSpeculativeGenerationStrategy(
            checkpoint_path=generate_args.ee_checkpoint, threshold=generate_args.ee_threshold,
            classifier_start_layer=generate_args.ee_classifier_start_layer,
            gpu_branching=generate_args.ee_gpu_branching,
        )
        strategy.prepare_model(model)
        return strategy
    if config.generation_strategy == "self_speculative":
        return SelfSpeculativeGenerationStrategy(cuda_graphs=generate_args.ss_cuda_graphs)
    if config.generation_strategy == "autoregressive":
        return AutoRegressiveGenerationStrategy()
    raise ValueError(f"Unsupported generation strategy: {config.generation_strategy}")


def _mean(values):
    return statistics.mean(values) if values else None


def summarize_runs(runs):
    tokens = sum(row["num_tokens_generated"] for row in runs)
    seconds = sum(row["total_time"] for row in runs)
    layers = sum(row["total_layer_executions"] for row in runs)
    phases = empty_layer_execution_counts()
    for row in runs:
        for phase, count in row["layer_executions_by_phase"].items():
            phases[phase] += count
    exit_counts = [row["generation_strategy_result"]["draft_exit_counts"] for row in runs]
    return {
        "num_examples": len(runs),
        "num_scored_examples": sum(row["num_tokens_generated"] > 0 for row in runs),
        "num_empty_generations": sum(row["num_tokens_generated"] == 0 for row in runs),
        "total_tokens": tokens, "total_generation_seconds": seconds,
        "tokens_per_second": tokens / seconds if seconds else None,
        "time_per_token": seconds / tokens if tokens else None,
        "total_layer_executions": layers, "layer_executions_by_phase": phases,
        "layers_per_token": layers / tokens if tokens else None,
        "total_classifier_calls": sum(row["total_classifier_calls"] for row in runs),
        "draft_exit_statistics": summarize_draft_exits([sum(counts) for counts in zip(*exit_counts)]),
    }


def benchmark(model, tokenizer, examples, bench, generate_args, config, seed):
    generator = HuggingfaceLlamaGenerator(tokenizer, model, build_strategy(model, generate_args, config))
    for index in range(bench.warmup):
        print(f"Warmup {index + 1}/{bench.warmup}", flush=True)
        generator.generate(prompt=examples[0].input, generation_config=config)

    metrics = TextMetrics()
    runs = []
    for index, example in enumerate(tqdm(examples, desc=config.generation_strategy)):
        # Independent of warmup and random numbers consumed by earlier prompts.
        example_seed = seed + index
        random.seed(example_seed)
        torch.manual_seed(example_seed)
        response = generator.generate(prompt=example.input, generation_config=config)
        row = {"example_id": index, "seed": example_seed, "prompt": example.input,
               "reference": example.output, **asdict(response)}
        runs.append(row)
        if response.num_tokens_generated:
            metrics.update(response.decoded_prediction, example.output)
        if bench.print_examples:
            print(f"[Prompt]:\n{example.input}\n[Reference Response]:\n{example.output}\n"
                  f"[Model Response]:\n{response.decoded_prediction}", flush=True)

    scored = [row for row in runs if row["num_tokens_generated"] > 0]
    acceptance = [row["generation_strategy_result"]["acceptance_rate"] for row in scored
                  if row["generation_strategy_result"]["acceptance_rate"] is not None]
    metric_result = {
        "predicted_text": metrics.compute(),
        "acceptance_rate": {"mean": _mean(acceptance)},
        **{key: {"mean": _mean([row[key] for row in scored])}
           for key in ("total_time", "time_per_token", "tokens_per_second")},
    }
    return metric_result, summarize_runs(runs), runs


def main(args, bench, generate_args, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    # Validate output destinations before creating input files or loading a model.
    output = Path(bench.output) if bench.output else Path(args.output_dir) / f"benchmark2_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    if not bench.prepare_only:
        for input_path in (bench.evaluation_file, bench.data_path):
            if input_path and output.resolve() == Path(input_path).resolve():
                raise ValueError("Report output must differ from the evaluation and source data files.")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Report already exists: {output}. Choose a new --output.")
    examples, digest = load_evaluation_set(bench, args.seed)
    print(f"Evaluation inputs SHA256: {digest}", flush=True)
    if bench.prepare_only:
        return {"num_examples": len(examples), "inputs_sha256": digest}
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    model.config.max_position_embeddings = 4096

    print("model.config.max_position_embeddings:", model.config.max_position_embeddings)

    metrics, summary, runs = benchmark(model, tokenizer, examples, bench, generate_args, config, args.seed)
    report = {
        "arguments": asdict(args), "benchmark_arguments": asdict(bench),
        "generate_arguments": asdict(generate_args), "generation_config": asdict(config),
        "evaluation": {"file": bench.evaluation_file, "num_examples": len(examples), "inputs_sha256": digest},
        "metric_result": metrics, "summary": summary, "runs": runs,
        "timing_scope": "Generation only, including prefill; excludes model loading, tokenization, decoding, graph setup, and warmup.",
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"metric_result": metrics, "summary": summary}, indent=2, allow_nan=False))
    if summary["draft_exit_statistics"] is not None:
        print(format_draft_exit_statistics(summary["draft_exit_statistics"]))
    print(report["timing_scope"])
    print(f"Saved report to {output}", flush=True)
    return report


if __name__ == "__main__":
    main(*process_cli_arguments())
