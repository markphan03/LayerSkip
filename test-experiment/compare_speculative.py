"""Compare fixed/adaptive speculation with the same graph and verification backend.

Example:
    python compare_speculative.py --model facebook/layerskip-llama2-7B \
        --exit-layer 8 --num-speculations 6 --max-new-tokens 64 --repeats 5
"""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_speculation.early_exit_speculation_generator import EarlyExitSelfSpeculativeGenerationStrategy
from self_speculation.generator_base import GenerationConfig
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy


def synchronize(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


@torch.inference_mode()
def measure_strategy(model, strategy, prompt_ids, eos_ids, config, warmup, repeats, seed):
    """Setup is separate; each timed run includes prefill and full verification."""
    synchronize(model)
    start = time.perf_counter()
    strategy.prepare_for_generation(model, prompt_ids, config)
    synchronize(model)
    setup_seconds = time.perf_counter() - start
    for index in range(warmup):
        torch.manual_seed(seed + index)
        strategy.generate_token_ids(model, prompt_ids, eos_ids, config)
    synchronize(model)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    samples = []
    for index in range(repeats):
        torch.manual_seed(seed + index)
        synchronize(model)
        start = time.perf_counter()
        result = strategy.generate_token_ids(model, prompt_ids, eos_ids, config)
        synchronize(model)
        elapsed = time.perf_counter() - start
        count = len(result.predicted_tokens)
        samples.append({
            "seconds": elapsed, "tokens": count, "tokens_per_second": count / elapsed,
            "acceptance_rate": result.acceptance_rate, "token_ids": result.predicted_tokens,
        })
    return {
        "config": asdict(config), "setup_seconds": setup_seconds,
        "median_seconds": statistics.median(row["seconds"] for row in samples),
        "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in samples),
        "mean_acceptance_rate": statistics.mean(row["acceptance_rate"] for row in samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(model.device) / 2**20 if model.device.type == "cuda" else None,
        "reserved_mib": torch.cuda.memory_reserved(model.device) / 2**20 if model.device.type == "cuda" else None,
        "samples": samples,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/layerskip-llama2-7B")
    parser.add_argument("--ee-checkpoint", default=None)
    parser.add_argument("--prompt", action="append", help="Repeat this option to benchmark several prompts.")
    parser.add_argument("--exit-layer", type=int, default=8, help="Fixed draft depth.")
    parser.add_argument("--adaptive-exit-layer", type=int, default=-1, help="Adaptive depth cap; -1 allows all layers.")
    parser.add_argument("--ee-threshold", type=float, default=.5)
    parser.add_argument("--ee-initial-fraction", type=float, default=.25)
    parser.add_argument("--num-speculations", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true", help="Default is greedy; sampled token sequences need not match.")
    parser.add_argument("--temperature", type=float, default=.6)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=.9)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report, including per-run measurements and tokens.")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_new_tokens < 1 or args.warmup < 0 or args.num_speculations < 0:
        parser.error("repeats and max-new-tokens must be positive; warmup and num-speculations must be non-negative.")
    if args.exit_layer < 1 or (args.adaptive_exit_layer != -1 and args.adaptive_exit_layer < 1):
        parser.error("exit-layer must be positive; adaptive-exit-layer must be -1 or positive.")
    return args


@torch.inference_mode()
def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for a portable-only comparison.")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=args.local_files_only,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    modes = ["off", "auto"] if args.device == "cuda" else ["off"]
    report = {
        "model": args.model, "torch": torch.__version__, "device": str(model.device),
        "gpu": torch.cuda.get_device_name(model.device) if args.device == "cuda" else None,
        "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
        "ee_checkpoint": args.ee_checkpoint or "checkpoints/ee_cnn/best_model/final.pt",
        "ee_threshold": args.ee_threshold, "ee_initial_fraction": args.ee_initial_fraction,
        "results": [],
    }
    for prompt in args.prompt or ["Question: Why is the sky blue?\nAnswer:"]:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_ids = encoded.input_ids[0].tolist()
        reference = None
        if not args.sample:
            reference = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(prompt_ids):].tolist()
            target_eos = model.generation_config.eos_token_id
            target_eos = target_eos if isinstance(target_eos, (tuple, list)) else [target_eos]
            for index, token in enumerate(reference):
                if token in set(eos_ids + target_eos):
                    reference = reference[:index]
                    break
        print(f"\nPrompt: {prompt!r}")
        print(f"{'Strategy':<14} {'Graphs':<6} {'Cap':>4} {'Setup s':>9} {'Median s':>9} {'Tokens/s':>9} {'Accept':>8} {'Reserved MiB':>13}")
        for adaptive in (False, True):
            for mode in modes:
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                strategy = (
                    EarlyExitSelfSpeculativeGenerationStrategy(
                        args.ee_checkpoint, args.ee_threshold, args.ee_initial_fraction, gpu_branching=mode,
                    ) if adaptive else SelfSpeculativeGenerationStrategy(cuda_graphs=mode)
                )
                depth = args.adaptive_exit_layer if adaptive else args.exit_layer
                config = GenerationConfig(
                    max_steps=args.max_new_tokens, exit_layer=depth, num_speculations=args.num_speculations,
                    sample=args.sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    generation_strategy="early_exit_speculative" if adaptive else "self_speculative",
                )
                row = measure_strategy(model, strategy, prompt_ids, eos_ids, config, args.warmup, args.repeats, args.seed)
                if reference is not None:
                    if any(sample["token_ids"] != reference for sample in row["samples"]):
                        raise AssertionError(f"Greedy output differs from full-model generation: adaptive={adaptive}, graphs={mode}.")
                row.update(prompt=prompt, prompt_tokens=len(prompt_ids), graphs=mode, greedy_matches_target=True if reference is not None else None)
                report["results"].append(row)
                memory = f"{row['reserved_mib']:.0f}" if row["reserved_mib"] is not None else "n/a"
                label = "adaptive" if adaptive else "fixed"
                print(f"{label:<14} {mode:<6} {depth:>4} {row['setup_seconds']:>9.3f} {row['median_seconds']:>9.3f} {row['median_tokens_per_second']:>9.2f} {row['mean_acceptance_rate']:>8.1%} {memory:>13}", flush=True)
                del strategy
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nSaved {args.output}")
    return report


if __name__ == "__main__":
    main(parse_args())
