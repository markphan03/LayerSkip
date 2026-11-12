# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple
from enum import Enum
from dataclasses import dataclass, field

import colorama
import random
import sys
import torch
import traceback
import transformers
import os

from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.early_exit_speculation_generator import EarlyExitSelfSpeculativeGenerationStrategy
from self_speculation.speculative_streamer import SpeculativeTextStreamer

class StreamerType(str, Enum):
    NONE="none"
    STANDARD="standard"
    SPECULATIVE="speculative"

@dataclass
class GenerateArguments:
    streamer: StreamerType = StreamerType.STANDARD
    ss_cuda_graphs: str = field(default="auto", metadata={
        "choices": ["auto", "off"],
        "help": "Fixed-depth self_speculative: auto uses shared-pool CUDA graphs on CUDA; off disables graphs.",
    })
    ee_checkpoint: str = field(default=None, metadata={
        "help": "GRU checkpoint for early_exit_speculative (default: checkpoints/ee_cnn/best_model/final.pt).",
    })
    ee_threshold: float = field(default=0.5, metadata={
        "help": "Exit at the first scheduled GRU check >= this value. exit_layer optionally caps draft depth.",
    })
    ee_initial_fraction: float = field(default=0.25, metadata={
        "help": "First exit check as a fraction of total layers, in (0, 1]. Subsequent increments halve, rounding up to at least one layer.",
    })
    ee_gpu_branching: str = field(default="auto", metadata={
        "choices": ["auto", "on", "off"],
        "help": "auto uses shared-pool CUDA graphs with grouped host checks; on opts into experimental GPU conditional graphs (high memory on affected PyTorch builds); off disables graphs.",
    })

def setup(args: Arguments, device: str = "cuda"):
    """Seed single-process generation; torchrun workers other than rank 0 exit."""
    # Generation uses no distributed operations, so plain Python needs neither
    # a process group nor rendezvous environment variables. Prefer global rank
    # when torchrun launches workers across multiple nodes.
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        raise SystemExit(0)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_model_and_tokenizer(args: Arguments, device: str = "auto"):
    local_model_path: str = args.model

    # initialize model
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        local_model_path,
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer

def main(args: Arguments, generate_arguments: GenerateArguments, generation_config: GenerationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    streamer = None
    match generate_arguments.streamer:
        case StreamerType.NONE:
            streamer = None
        case StreamerType.STANDARD:
            streamer = transformers.TextStreamer(tokenizer)
        case StreamerType.SPECULATIVE:
            streamer = SpeculativeTextStreamer(tokenizer)
        case _:
            raise ValueError(f"Unsupported streamer type {generate_arguments.streamer}")

    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy(
            cuda_graphs=generate_arguments.ss_cuda_graphs,
        )
    elif generation_config.generation_strategy == "early_exit_speculative":
        generation_strategy = EarlyExitSelfSpeculativeGenerationStrategy(
            checkpoint_path=generate_arguments.ee_checkpoint,
            threshold=generate_arguments.ee_threshold,
            initial_fraction=generate_arguments.ee_initial_fraction,
            gpu_branching=generate_arguments.ee_gpu_branching,
        )
        generation_strategy.prepare_model(model)
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    # Warmup
    warmup = 1
    for _ in range(warmup):
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generate(**tokenizer("This is a warmup prompt", return_tensors="pt").to(device), max_new_tokens=10)

    while True:
        print()
        print("Enter a prompt and then press ctrl+d twice for the model to complete:")
        print("======================================================================")
        print()

        print(colorama.Fore.LIGHTBLUE_EX, end="")
        prompt=sys.stdin.read()
        print(colorama.Style.RESET_ALL, end=" ")

        # An empty read indicates EOF; stop instead of generating repeatedly.
        if not prompt.strip():
            break

        try:
            response: GenerationResult = generator.generate(
                prompt=prompt,
                generation_config=generation_config,
                streamer=streamer,
            )
        except:
            print(colorama.Style.RESET_ALL)
            traceback.print_exc()
            raise
        num_tokens = response.num_tokens_generated
        total_time = response.total_time

        if streamer:
            streamer.end()
        else:
            print(response.decoded_prediction)

        print(colorama.Style.RESET_ALL)
        print()
        print(f"\tTime taken: {total_time :.3f}s")
        print(f"\tNumber of tokens: {num_tokens}")
        print(f"\tTime per token: {total_time / num_tokens if num_tokens else 0 : .3f}s")
        print(f"\tTokens per second: {num_tokens / total_time :.3f}")
        if generation_config.generation_strategy in ("self_speculative", "early_exit_speculative"):
            print(f"\tAcceptance Rate: {response.generation_strategy_result.acceptance_rate:.2%}")
        print()

def process_cli_arguments(argv=None) -> Tuple[Arguments, GenerateArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((Arguments, GenerateArguments, GenerationConfig))
    print(parser)

    (
        general_arguments,
        generate_arguments,
        generation_config,
    ) = parser.parse_args_into_dataclasses(args=argv, return_remaining_strings=False)

    if generation_config.generation_strategy == "early_exit_speculative":
        if not 0 <= generate_arguments.ee_threshold <= 1:
            parser.error("--ee_threshold must be between 0 and 1.")
        if not 0 < generate_arguments.ee_initial_fraction <= 1:
            parser.error("--ee_initial_fraction must be greater than 0 and at most 1.")
        if generation_config.num_speculations < 0:
            parser.error("--num_speculations must be non-negative for early_exit_speculative.")
        if generation_config.exit_layer != -1 and generation_config.exit_layer < 1:
            parser.error("--exit_layer must be -1 or a positive maximum draft depth.")

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, generate_arguments, generation_config

if __name__ == "__main__":
    print(sys.argv[:])
    args, benchmark_arguments, generation_config = process_cli_arguments()
    print(type(args), args)
    print(type(benchmark_arguments), benchmark_arguments)
    print(type(generation_config), generation_config)
    main(args, benchmark_arguments, generation_config)
