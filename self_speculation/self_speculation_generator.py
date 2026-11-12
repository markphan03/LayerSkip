# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple

import colorama
import torch

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
)

def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)

class SelfSpeculativeGenerationStrategy(GenerationStrategy):
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
        past_key_values = None

        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_ids=eos_token_ids,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    # break out of loop when we get an EOS token
                    # remove the EOS token id
                    output_ids = output_ids[: output_ids.index(eos_token_id)]
                    eos_found = True
                    break
            if eos_found:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations,
        )

    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None
        for _ in range(num_speculations):
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)
            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token in eos_token_ids:
                # break out of loop when we get an EOS token
                break

        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat(
            [input_ids, draft_output_ids],
            dim=-1,
        )

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)

        # logits: 1 x (T_d  + T_p) x V
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)

        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                # streamer.put(torch.cat((draft_output_ids[0, : number_of_matches], verified_tokens[0][number_of_matches : number_of_matches + 1])))
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )


@torch.inference_mode()
def generate_with_early_exit(
    question: str,
    checkpoint_path=None,
    *,
    model=None,
    tokenizer=None,
    model_path=None,
    max_new_tokens: int = 128,
    threshold: float = 0.5,
    sample: bool = False,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    device=None,
) -> str:
    """Answer one question using the trained GRU's first accepted decoder layer.

    Example::

        answer = generate_with_early_exit("Why is the sky blue?", max_new_tokens=64)

    The default checkpoint is checkpoints/ee_cnn/best_model/final.pt relative
    to the repository. The base model/tokenizer are loaded from its training
    metadata unless supplied; pass both to reuse an already loaded Llama.
    They must match the checkpoint's training model and reside on one device.
    ``device`` selects placement when loading a model. Supplied models retain
    their placement. Models are put in evaluation mode.

    Chat tokenizers use their chat template; otherwise the prompt is
    "Question: <question>\\nAnswer:". Only the generated answer is returned.
    This classifier was trained on CNN/DailyMail summaries, so question
    answering quality is not established by its training metrics.

    Raw last-token decoder states feed the GRU across layers, resetting its
    state for each new token. A sigmoid score >= threshold selects that layer;
    otherwise the final layer is used. Selected states pass through Llama's
    final norm and vocabulary head. Tokens are not speculatively verified.

    Each step recomputes the entire prefix without a KV cache, then skips layers
    after the chosen exit. This preserves attention context when exit depth
    changes between tokens; it is not an optimized decoding implementation.
    """
    import inspect
    from pathlib import Path

    from train_ee_cnn import EarlyExitGRU

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string.")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if sample and not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive when sampling.")
    if not isinstance(top_k, int) or top_k < 0 or not 0 <= top_p <= 1:
        raise ValueError("top_k must be a non-negative integer and top_p between 0 and 1.")
    if (model is None) != (tokenizer is None):
        raise ValueError("Supply both model and tokenizer, or neither.")
    if max_new_tokens == 0:
        return ""

    if checkpoint_path is None:
        checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / "checkpoints/ee_cnn/best_model/final.pt"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classifier = EarlyExitGRU(**checkpoint["model_config"])
    classifier.load_state_dict(checkpoint["model_state_dict"])
    if model is None:
        model_path = model_path or checkpoint.get("training_args", {}).get("model_path")
        if not model_path:
            raise ValueError("The checkpoint has no training model_path; supply model_path.")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device)
    del checkpoint

    if not isinstance(model, transformers.LlamaForCausalLM):
        raise ValueError("generate_with_early_exit requires a LlamaForCausalLM.")
    backbone = model.model
    device = backbone.embed_tokens.weight.device
    if device.type == "meta" or any(p.device != device for p in model.parameters()):
        raise ValueError("The Llama model must reside entirely on one device.")
    if backbone.embed_tokens.embedding_dim != classifier.config["input_size"]:
        raise ValueError("The checkpoint input_size does not match the Llama hidden size.")
    if not backbone.layers:
        raise ValueError("The Llama model must have at least one decoder layer.")
    model.eval()
    classifier = classifier.to(device).eval()
    classifier_dtype = next(classifier.parameters()).dtype

    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
    else:
        input_ids = tokenizer(
            f"Question: {question}\nAnswer:", return_tensors="pt", add_special_tokens=True,
        )["input_ids"]
    input_ids = input_ids.to(device)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("The tokenizer must return one nonempty token sequence.")
    if input_ids.shape[1] + max_new_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError("The question and requested answer exceed the model context length.")

    eos_token_ids = set()
    for value in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if value is not None:
            eos_token_ids.update(value if isinstance(value, (list, tuple)) else [value])
    layer_parameters = [inspect.signature(layer.forward).parameters for layer in backbone.layers]
    output_ids = []
    for _ in range(max_new_tokens):
        hidden_states = backbone.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = torch.full(
            (seq_len, seq_len), torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype, device=device,
        ).triu(diagonal=1)[None, None]
        position_embeddings = (
            backbone.rotary_emb(hidden_states, position_ids)
            if hasattr(backbone, "rotary_emb") else None
        )
        gru_state = None
        for layer_index, (layer, parameters) in enumerate(zip(backbone.layers, layer_parameters)):
            kwargs = {"attention_mask": causal_mask, "position_ids": position_ids}
            optional_kwargs = {
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
                "use_cache": False,
                "output_attentions": False,
            }
            kwargs.update({key: value for key, value in optional_kwargs.items() if key in parameters})
            layer_output = layer(hidden_states, **kwargs)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            raw = hidden_states[:, -1:, :]
            if layer_index == len(backbone.layers) - 1:
                break
            states, gru_state = classifier.gru(raw.to(dtype=classifier_dtype), gru_state)
            score = classifier.head(states[:, 0]).sigmoid().item()
            if not 0 <= score <= 1:
                raise ValueError("The early-exit classifier produced a non-finite score.")
            if score >= threshold:
                break

        logits = model.lm_head(backbone.norm(raw))
        next_token, _ = decode_next_token(
            logits, token_idx=-1, sample=sample, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
        token_id = next_token.item()
        if token_id in eos_token_ids:
            break
        output_ids.append(token_id)
        input_ids = torch.cat([input_ids, input_ids.new_tensor([[token_id]])], dim=-1)
    return tokenizer.decode(output_ids, skip_special_tokens=True)
