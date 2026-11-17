# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

import transformers

from self_speculation.exit_statistics import summarize_draft_exits


def empty_layer_execution_counts():
    return dict(prefill=0, draft=0, verification=0, decode=0)


def format_layer_execution_counts(counts):
    phases = [f"{phase.capitalize()}: {counts[phase]}" for phase in ("prefill", "draft", "verification")]
    if counts["decode"]:
        phases.append(f"Decode: {counts['decode']}")
    return "Layer executions by phase | " + " | ".join(phases)


@dataclass
class GenerationCounters:
    """Actual decoder work, including prefill and discarded speculation.

    A layer execution is one decoder-layer call. A classifier call
    is one GRU/head exit decision (not a vocabulary projection).
    """

    total_layer_executions: int = 0
    total_classifier_calls: int = 0
    layer_executions_by_phase: Dict[str, int] = field(default_factory=empty_layer_execution_counts)

    def add(self, other):
        self.total_layer_executions += other.total_layer_executions
        self.total_classifier_calls += other.total_classifier_calls
        for phase, count in other.layer_executions_by_phase.items():
            self.layer_executions_by_phase[phase] += count

    def record_layers(self, layers, phase="decode"):
        self.layer_executions_by_phase[phase] += layers
        self.total_layer_executions += layers


@dataclass
class GenerationStrategyResult:
    predicted_tokens: List[int]
    acceptance_rate: Optional[float] = None
    total_layer_executions: int = 0
    total_classifier_calls: int = 0
    draft_exit_counts: List[int] = field(default_factory=list)
    layer_executions_by_phase: Dict[str, int] = field(default_factory=empty_layer_execution_counts)

    @property
    def num_tokens_generated(self):
        return len(self.predicted_tokens)

    @property
    def layers_per_token(self):
        return self.total_layer_executions / self.num_tokens_generated if self.num_tokens_generated else None


@dataclass
class GenerationResult:
    generation_strategy_result: GenerationStrategyResult
    decoded_prediction: str
    num_tokens_generated: int
    total_time: float
    time_per_token: float
    tokens_per_second: float
    total_layer_executions: int = field(init=False)
    total_classifier_calls: int = field(init=False)
    layers_per_token: Optional[float] = field(init=False)
    draft_exit_statistics: Optional[dict] = field(init=False)
    layer_executions_by_phase: Dict[str, int] = field(init=False)

    def __post_init__(self):
        result = self.generation_strategy_result
        self.total_layer_executions = result.total_layer_executions
        self.total_classifier_calls = result.total_classifier_calls
        self.layers_per_token = result.layers_per_token
        self.draft_exit_statistics = summarize_draft_exits(result.draft_exit_counts)
        self.layer_executions_by_phase = result.layer_executions_by_phase.copy()


@dataclass
class GenerationConfig:
    max_steps: int = 512
    exit_layer: int = -1
    num_speculations: int = -1
    generation_strategy: str = "autoregressive"
    sample: bool = True
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 0.9
    no_repeat_ngram_size: int = None
    stop_words: List[str] = None
    stop_token_ids: List[int] = None

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []

class GenerationStrategy:
    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,  
    ) -> GenerationStrategyResult:
        raise NotImplementedError()


class HuggingfaceLlamaGenerator:
    def __init__(
        self,
        tokenizer: transformers.LlamaTokenizer,
        model: transformers.LlamaForCausalLM,
        generation_strategy: GenerationStrategy,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.generation_strategy = generation_strategy

    def create_logits_processors(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.generation.logits_process.LogitsProcessorList:
        logits_processors: transformers.generation.logits_process.LogitsProcessorList = transformers.generation.logits_process.LogitsProcessorList()
        if generation_config.no_repeat_ngram_size:
            logits_processors.append(transformers.generation.logits_process.NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))

        return logits_processors

    def create_stopping_criteria(
            self,
            generation_config: GenerationConfig,
    ) -> transformers.StoppingCriteriaList:
        stopping_criteria: transformers.StoppingCriteriaList = transformers.StoppingCriteriaList()
        if generation_config.stop_words:
            stopping_criteria.append(transformers.StopStringCriteria(self.tokenizer, generation_config.stop_words))

        return stopping_criteria

    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationResult:
        example = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        logits_processors = self.create_logits_processors(generation_config=generation_config)
        stopping_criteria = self.create_stopping_criteria(generation_config)
        eos_token_ids = generation_config.stop_token_ids + [self.tokenizer.eos_token_id]
        # Optional strategy setup (e.g. CUDA graph capture) is not token latency.
        prepare = getattr(self.generation_strategy, "prepare_for_generation", None)
        if prepare is not None:
            prepare(self.model, example["input_ids"].tolist()[0], generation_config)
        cuda_devices = {parameter.device for parameter in self.model.parameters() if parameter.device.type == "cuda"}
        with torch.inference_mode():
            for device in cuda_devices:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            generation_strategy_result = self.generation_strategy.generate_token_ids(
                model=self.model,
                input_ids=example["input_ids"].tolist()[0],
                eos_token_ids=eos_token_ids,
                generation_config=generation_config,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            for device in cuda_devices:
                torch.cuda.synchronize(device)
            total_time = time.perf_counter() - start
        decoded_prediction = self.tokenizer.decode(
            generation_strategy_result.predicted_tokens
        )
        num_tokens_generated = len(generation_strategy_result.predicted_tokens)
        return GenerationResult(
            generation_strategy_result=generation_strategy_result,
            decoded_prediction=decoded_prediction,
            num_tokens_generated=num_tokens_generated,
            total_time=total_time,
            time_per_token=total_time / num_tokens_generated if num_tokens_generated > 0 else None,
            tokens_per_second=num_tokens_generated / total_time,
        )
