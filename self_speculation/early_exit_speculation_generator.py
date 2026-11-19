"""GRU-selected draft depths with full-model speculative verification.

Unlike unverified early-exit generation, the full Llama remains the target
distribution. Greedy verification accepts matching tokens; sampling uses the
acceptance ratio p(token)/q(token) and the positive residual (p - q) on rejection.
"""

from dataclasses import asdict
from pathlib import Path
import time

import torch
import transformers
from transformers.cache_utils import DynamicCache

from self_speculation.generator_base import GenerationCounters, GenerationStrategy, GenerationStrategyResult
from self_speculation.llama_model_utils import top_k_top_p_filtering
from self_speculation.speculative_streamer import SpeculativeTextStreamer


def _validate_classifier_start_layer(classifier_start_layer):
    if type(classifier_start_layer) is not int or classifier_start_layer < 1:
        raise ValueError("--ee-classifier-start-layer must be a positive integer (1-based).")


def exit_check_layers(num_layers, max_depth=None, classifier_start_layer=1):
    """Check consecutive layers from a 1-based start, ending at the forced cap.

    The cap is always included, even when it precedes the first classifier
    check. No classifier decision is needed at the cap itself.
    """
    _validate_classifier_start_layer(classifier_start_layer)
    if classifier_start_layer > num_layers:
        raise ValueError("--ee-classifier-start-layer exceeds the model's number of layers.")
    max_depth = num_layers if max_depth is None else max_depth
    if not 1 <= max_depth <= num_layers:
        raise ValueError("max_depth must be between 1 and num_layers.")
    return list(range(min(classifier_start_layer, max_depth), max_depth + 1))


class _AdaptiveDraftCache:
    """Catch up skipped layers without recomputing the verified prefix.

    At the start of a round every KV layer has ``base_length`` entries. During
    drafting they may have different lengths. ``states[i]`` retains the raw
    inputs to layer i for this round, so a deeper exit can process all missing
    positions. Verification catches up every layer before a shared rollback.
    """

    def __init__(self, model, cache, input_ids):
        self.model = model
        self.cache = cache
        self.base_length = cache.get_seq_length()
        self.states = [model.model.embed_tokens(input_ids)] + [None] * len(model.model.layers)
        self.position_data = {}
        self.counters = GenerationCounters()

    def append(self, token):
        ids = torch.tensor([[token]], device=self.states[0].device)
        self.states[0] = torch.cat([self.states[0], self.model.model.embed_tokens(ids)], dim=1)
        self.position_data.clear()

    def _run_layer(self, index):
        backbone = self.model.model
        start = self.cache.get_seq_length(index)
        end = self.base_length + self.states[0].shape[1]
        if start < end:
            hidden = self.states[index][:, start - self.base_length:]
            if start not in self.position_data:
                positions = torch.arange(start, end, device=hidden.device)
                blocked = torch.arange(end, device=hidden.device)[None, :] > positions[:, None]
                mask = hidden.new_zeros((1, 1, len(positions), end))
                mask.masked_fill_(blocked[None, None], torch.finfo(hidden.dtype).min)
                position_ids = positions.unsqueeze(0)
                embeddings = backbone.rotary_emb(hidden, position_ids)
                self.position_data[start] = positions, position_ids, mask, embeddings
            positions, position_ids, mask, embeddings = self.position_data[start]
            output = backbone.layers[index](
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                use_cache=True,
                output_attentions=False,
                cache_position=positions,
                position_embeddings=embeddings,
            )
            raw = output[0] if isinstance(output, tuple) else output
            self.counters.record_layers(1)
            previous = self.states[index + 1]
            self.states[index + 1] = raw if previous is None else torch.cat([previous, raw], dim=1)
        return self.states[index + 1]

    def draft_logits(self, classifier, threshold, max_depth, classifier_start_layer=1):
        state = None  # GRU sequences run across layers, independently per token.
        start = 0
        checks = (exit_check_layers(len(self.model.model.layers), max_depth, classifier_start_layer)
                  if classifier is not None else [max_depth])
        for depth in checks:
            raw_states = [self._run_layer(index)[:, -1:] for index in range(start, depth)]
            raw = raw_states[-1]
            if depth == max_depth:
                break
            # Preserve the training sequence across ALL decoder layers, while
            # issuing one GRU call and one host decision per eligible layer.
            states, state = classifier.gru(torch.cat(raw_states, dim=1), state)
            score = classifier.head(states[:, -1]).sigmoid().item()
            self.counters.total_classifier_calls += 1
            if not 0 <= score <= 1:
                raise ValueError("The early-exit classifier produced a non-finite score.")
            if score >= threshold:
                break
            start = depth
        logits = self.model.lm_head(self.model.model.norm(raw))[:, 0]
        return logits, depth

    def verification_logits(self, count):
        for index in range(len(self.model.model.layers)):
            raw = self._run_layer(index)
        # Project only draft verification positions and the bonus position.
        return self.model.lm_head(self.model.model.norm(raw[:, -count:]))


def _processed_scores(logits, prefix, config, logits_processors):
    if not config.sample and not logits_processors:
        return logits
    scores = logits.float().clone()
    if logits_processors:
        scores = logits_processors(prefix, scores)
    if config.sample:
        scores = top_k_top_p_filtering(
            scores / config.temperature, top_k=config.top_k, top_p=config.top_p,
        )
    return scores


def _verify_sample(token, target, draft):
    """Return (accepted or corrected token, accepted) for normalized p and q."""
    ratio = (target[0, token] / draft[0, token]).clamp(max=1)
    if torch.rand((), device=target.device) < ratio:
        return token, True
    residual = (target - draft).clamp_min(0)
    mass = residual.sum()
    # Exact rejection implies positive residual mass; tolerate roundoff.
    probabilities = residual / mass if mass > 0 else target
    return torch.multinomial(probabilities, num_samples=1).item(), False


class VerifiedSpeculativeGenerationStrategy(GenerationStrategy):
    """Shared cache, graph setup, and exact verification for both draft policies.

    With no classifier, drafting stops at the configured
    fixed depth. Adaptive subclasses supply a classifier and an exit schedule.
    """

    def __init__(self, classifier=None, threshold=0.5, classifier_start_layer=1, gpu_branching="auto"):
        if not 0 <= threshold <= 1:
            raise ValueError("ee_threshold must be between 0 and 1.")
        _validate_classifier_start_layer(classifier_start_layer)
        if gpu_branching not in ("auto", "on", "off"):
            raise ValueError("ee_gpu_branching must be auto, on, or off.")
        self.classifier = classifier
        self.threshold = threshold
        self.classifier_start_layer = classifier_start_layer
        self.gpu_branching = gpu_branching
        self._cuda_graphs = None
        self._cuda_key = None
        self.graph_setup_seconds = 0.0

    def prepare_model(self, model):
        """Match the base model before warmup or timed generation."""
        parameter = model.model.embed_tokens.weight
        if self.classifier is not None:
            classifier_parameter = next(self.classifier.parameters())
            if classifier_parameter.device != parameter.device or classifier_parameter.dtype != parameter.dtype:
                self.classifier.to(device=parameter.device, dtype=parameter.dtype)
            self.classifier.eval()
        model.eval()

    @torch.inference_mode()
    def prepare_for_generation(self, model, input_ids, generation_config):
        """Build reusable CUDA graphs outside HuggingfaceLlamaGenerator's timer."""
        from self_speculation.early_exit_cuda_graphs import (
            ConditionalDraftGraphs, GroupedDraftGraphs, conditional_graphs_available,
        )

        config = generation_config
        if not isinstance(model, transformers.LlamaForCausalLM):
            raise ValueError("Speculative decoding requires LlamaForCausalLM.")
        parameter = model.model.embed_tokens.weight
        if parameter.device.type == "meta" or any(p.device != parameter.device for p in model.parameters()):
            raise ValueError("Speculative decoding requires the model on one device.")
        if self.classifier is not None and model.config.hidden_size != self.classifier.config["input_size"]:
            raise ValueError("The checkpoint input_size does not match the Llama hidden size.")
        depth = len(model.model.layers) if config.exit_layer == -1 else config.exit_layer
        checks = (exit_check_layers(len(model.model.layers), depth, self.classifier_start_layer)
                  if self.classifier is not None else [depth])
        self.prepare_model(model)
        available = conditional_graphs_available(model)
        if self.gpu_branching == "on" and not available:
            raise RuntimeError(
                "--ee_gpu_branching on needs CUDA and the CUDAGraph conditional-node API. "
                "Use auto/off for grouped host checks on other systems."
            )
        if self.gpu_branching == "off" or parameter.device.type != "cuda" or config.max_steps == 0:
            self._cuda_graphs = None
            self._cuda_key = None
            return
        if model.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("CUDA exit graphs require eager or sdpa attention.")
        if config.max_steps < 0 or config.num_speculations < 0:
            raise ValueError("max_steps and num_speculations must be non-negative.")
        if not input_ids:
            raise ValueError("input_ids must contain one nonempty prompt.")
        required = len(input_ids) + config.max_steps - 1
        if required > model.config.max_position_embeddings:
            raise ValueError("The prompt and max_steps exceed the model context length. prepare_for_generation")
        capacity = min(((required + 127) // 128) * 128, model.config.max_position_embeddings)
        max_drafts = min(config.num_speculations, config.max_steps - 1)
        key = (id(model), parameter.device, parameter.dtype, capacity, max_drafts, tuple(checks), self.threshold, self.gpu_branching)
        if key != self._cuda_key:
            # Release old graphs before allocating their replacement.
            self._cuda_graphs = None
            self._cuda_key = None
            start = time.perf_counter()
            engine = ConditionalDraftGraphs if self.gpu_branching == "on" else GroupedDraftGraphs
            self._cuda_graphs = engine(
                model, self.classifier, checks, self.threshold, capacity, max_drafts,
            )
            torch.cuda.synchronize(parameter.device)
            self.graph_setup_seconds = time.perf_counter() - start
            self._cuda_key = key

    @torch.inference_mode()
    def generate_token_ids(
        self, model, input_ids, eos_token_ids, generation_config,
        logits_processors=None, stopping_criteria=None, streamer=None,
    ):
        config = generation_config
        if config.max_steps < 0 or config.num_speculations < 0:
            raise ValueError("max_steps and num_speculations must be non-negative.")
        if config.sample and not 0 < config.temperature < float("inf"):
            raise ValueError("temperature must be finite and positive when sampling.")
        if config.top_k < 0 or not 0 <= config.top_p <= 1:
            raise ValueError("top_k must be non-negative and top_p between 0 and 1.")
        if not isinstance(model, transformers.LlamaForCausalLM):
            raise ValueError("Speculative decoding requires LlamaForCausalLM.")
        device = model.model.embed_tokens.weight.device
        if device.type == "meta" or any(p.device != device for p in model.parameters()):
            raise ValueError("Speculative decoding requires the model on one device.")
        if model.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Speculative decoding requires eager or sdpa attention.")
        if self.classifier is not None and model.config.hidden_size != self.classifier.config["input_size"]:
            raise ValueError("The checkpoint input_size does not match the Llama hidden size.")
        depth = len(model.model.layers) if config.exit_layer == -1 else config.exit_layer
        if not 1 <= depth <= len(model.model.layers):
            raise ValueError("exit_layer must be -1 or a valid 1-based maximum draft depth.")
        context = torch.tensor([input_ids], dtype=torch.long, device=device)
        if context.ndim != 2 or context.shape[1] == 0:
            raise ValueError("input_ids must contain one nonempty prompt.")
        if context.shape[1] + config.max_steps - 1 > model.config.max_position_embeddings:
            print(f"{context.shape[1]} + {config.max_steps - 1} > {model.config.max_position_embeddings}")
            raise ValueError("The prompt and max_steps exceed the model context length. generate_token_ids")
        self.prepare_for_generation(model, input_ids, config)
        eos = set(eos_token_ids or [])
        model_eos = model.generation_config.eos_token_id
        if model_eos is not None:
            eos.update(model_eos if isinstance(model_eos, (list, tuple)) else [model_eos])
        cache = DynamicCache()
        counters = GenerationCounters()
        if self._cuda_graphs is not None:
            self._cuda_graphs.prefill(context)
        elif config.max_steps and context.shape[1] > 1:
            # Prefill normally so the adaptive cache only retains the short
            # speculative suffix, rather than every layer's prompt activations.
            cache = model.model(context[:, :-1], use_cache=True).past_key_values
        if config.max_steps and context.shape[1] > 1:
            counters.record_layers(len(model.model.layers), "prefill")
        output_ids = []
        draft_exit_counts = [0] * len(model.model.layers)
        total_drafts = total_accepted = 0
        done = False

        def emit(token):
            nonlocal context
            if token in eos:
                return True
            output_ids.append(token)
            new_token = context.new_tensor([[token]])
            context = torch.cat([context, new_token], dim=1)
            if streamer is not None:
                streamer.put(new_token.cpu())
            return bool(stopping_criteria and torch.all(stopping_criteria(context, scores=None)))

        while len(output_ids) < config.max_steps and not done:
            round_cache = (
                self._cuda_graphs.begin_round(context) if self._cuda_graphs is not None
                else _AdaptiveDraftCache(model, cache, context[:, cache.get_seq_length():])
            )
            draft_context = context
            draft_tokens, draft_probabilities = [], []
            budget = min(config.num_speculations, config.max_steps - len(output_ids) - 1)
            for _ in range(budget):
                logits, exit_depth = round_cache.draft_logits(
                    self.classifier, self.threshold, depth, self.classifier_start_layer,
                )
                if self._cuda_graphs is None:
                    draft_exit_counts[exit_depth - 1] += 1
                scores = _processed_scores(logits, draft_context, config, logits_processors)
                if config.sample:
                    probabilities = scores.softmax(dim=-1)
                    token = torch.multinomial(probabilities, num_samples=1).item()
                    draft_probabilities.append(probabilities)
                else:
                    token = scores.argmax(dim=-1).item()
                draft_tokens.append(token)
                round_cache.append(token)
                draft_context = torch.cat([draft_context, context.new_tensor([[token]])], dim=1)
                if token in eos:
                    break
            total_drafts += len(draft_tokens)
            speculative_streamer = isinstance(streamer, SpeculativeTextStreamer)
            if speculative_streamer and draft_tokens:
                streamer.put(torch.tensor(draft_tokens), is_draft=True)
            verified = round_cache.verification_logits(len(draft_tokens) + 1)
            if self._cuda_graphs is None:
                counters.add(round_cache.counters)
            if speculative_streamer and draft_tokens:
                streamer.delete(len(draft_tokens))

            # Without prefix-dependent processors, greedy verification needs
            # only one device-to-host transfer for the whole block.
            greedy_tokens = (
                verified.argmax(dim=-1)[0].tolist()
                if not config.sample and not logits_processors else None
            )
            rejected = False
            for index, draft_token in enumerate(draft_tokens):
                # context includes only the prompt and tokens accepted so far.
                if greedy_tokens is not None:
                    token = greedy_tokens[index]
                    accepted = token == draft_token
                else:
                    scores = _processed_scores(verified[:, index], context, config, logits_processors)
                    if config.sample:
                        token, accepted = _verify_sample(
                            draft_token, scores.softmax(dim=-1), draft_probabilities[index],
                        )
                    else:
                        token = scores.argmax(dim=-1).item()
                        accepted = token == draft_token
                total_accepted += int(accepted)
                done = emit(token)
                rejected = not accepted
                if rejected or done:
                    break
            if not rejected and not done:
                if greedy_tokens is not None:
                    token = greedy_tokens[-1]
                else:
                    scores = _processed_scores(verified[:, -1], context, config, logits_processors)
                    token = (
                        torch.multinomial(scores.softmax(dim=-1), num_samples=1).item()
                        if config.sample else scores.argmax(dim=-1).item()
                    )
                done = emit(token)
            # Keep the verified prefix; the last emitted token is next round's input.
            if self._cuda_graphs is None:
                cache.crop(context.shape[1] - 1)

        if self._cuda_graphs is not None:
            counters.add(self._cuda_graphs.generation_counters())
            draft_exit_counts = self._cuda_graphs.generation_exit_counts()
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            draft_exit_counts=draft_exit_counts,
            **asdict(counters),
            acceptance_rate=total_accepted / total_drafts if total_drafts else 0.0,
        )


class EarlyExitSelfSpeculativeGenerationStrategy(VerifiedSpeculativeGenerationStrategy):
    """Draft at the first scheduled GRU score >= threshold, then fully verify.

    ``GenerationConfig.exit_layer`` optionally caps draft depth (1-based).
    Its default -1 allows all layers. The final allowed layer is the fallback.
    ``num_speculations`` is the maximum draft block length, including zero.
    The checkpoint's training model must be used on one device with eager or
    SDPA attention. The classifier is loaded once and reused across prompts.
    Classifier checks start at classifier_start_layer (1-based) and continue
    at every layer before the forced draft-depth cap. The GRU still consumes
    all preceding layer states. Every draft token gets an independent decision.
    CUDA auto mode uses ordinary graphs with host checks at each eligible layer.
    Explicit on mode opts into conditional graphs.
    """

    def __init__(
        self, checkpoint_path=None, threshold=0.5, classifier_start_layer=1,
        gpu_branching="auto",
    ):
        from train_ee_cnn import EarlyExitGRU

        if not 0 <= threshold <= 1:
            raise ValueError("ee_threshold must be between 0 and 1.")
        _validate_classifier_start_layer(classifier_start_layer)
        if gpu_branching not in ("auto", "on", "off"):
            raise ValueError("ee_gpu_branching must be auto, on, or off.")
        if checkpoint_path is None:
            checkpoint_path = (
                Path(__file__).resolve().parents[1] / "checkpoints/ee_cnn/best_model/final.pt"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.classifier = EarlyExitGRU(**checkpoint["model_config"])
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.eval()
        super().__init__(self.classifier, threshold, classifier_start_layer, gpu_branching)
