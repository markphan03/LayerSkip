"""Evaluate the trained early-exit GRU on CNN/DailyMail next-token prediction.

Run: python eval_ee_cnn.py --checkpoint checkpoints/ee_cnn/final.pt
Example: python eval_ee_cnn.py --max-documents 100 --threshold 0.5 --output ee_metrics.json
Tests: python -m pytest tests/test_eval_ee_cnn.py

The default split is test. Tokenization, teacher forcing, raw decoder states,
and next-token alignment match train_ee_cnn.py. Only reference-summary tokens
plus EOS are scored; article tokens and padding are excluded.

For each token, select the FIRST decoder layer whose GRU sigmoid score reaches
--threshold. If none does, use the final layer. Accuracy means equality with
the reference token, not the training label matches_reference_or_final.

This evaluates the exit policy using a full teacher-forced Llama pass. All
decoder layers still run to preserve attention context and obtain the final-
layer baseline. Mean exit depth describes the selected predictions, not actual
computation saved or measured decoding speed. No autoregressive text is generated.

Model and preprocessing defaults come from the classifier checkpoint. The test
split, starting row, and document limit are selected independently of training.
JSON accuracies are fractions in [0, 1]; terminal output displays percentages.
"""

import argparse
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_ee_cnn import DATASET, MODEL, SPLITS, VERSIONS, EarlyExitGRU, TokenEncoder, load_records


def predict_tokens(model, raw, projection_chunk_size):
    """Use the same final normalization and vocabulary head as training."""
    predictions = torch.empty(len(raw), dtype=torch.long, device=raw.device)
    for start in range(0, len(raw), projection_chunk_size):
        stop = start + projection_chunk_size
        logits = model.lm_head(model.model.norm(raw[start:stop]))
        predictions[start:stop] = logits.argmax(dim=-1)
        del logits
    return predictions


@torch.inference_mode()
def forward_layer_by_layer(
    model, classifier, input_ids, attention_mask, labels,
    threshold=0.5, projection_chunk_size=32,
):
    """Select exits using only states available at that layer; return CPU IDs.

The GRU state is independent for each target token and persists across decoder
layers, exactly as in EarlyExitGRU.forward. Exit layers are numbered from 1.
The Llama residual stream always receives its unnormalized decoder output.
"""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if projection_chunk_size < 1:
        raise ValueError("projection_chunk_size must be positive.")
    model.eval()
    classifier.eval()
    backbone = model.model
    device = backbone.embed_tokens.weight.device
    classifier_parameter = next(classifier.parameters())
    if classifier_parameter.device != device:
        raise ValueError("Llama and the classifier must be on the same device.")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # 1. Token IDs -> embeddings. Select the same source positions as training:
    #    the hidden state at position j predicts the reference token at j + 1.
    hidden_states = backbone.embed_tokens(input_ids)
    batch_size, seq_len, _ = hidden_states.shape
    target_mask = labels.ne(-100)
    batch_indices, target_positions = target_mask.nonzero(as_tuple=True)
    if (
        not target_mask.any(dim=1).all().item()
        or target_positions.eq(0).any().item()
    ):
        raise ValueError("Each document needs targets with preceding input tokens.")
    source_positions = target_positions - 1
    if not (
        attention_mask[batch_indices, target_positions].bool().all().item()
        and attention_mask[batch_indices, source_positions].bool().all().item()
    ):
        raise ValueError("Target tokens and their source positions must not be padding.")
    reference_ids = labels[batch_indices, target_positions]
    num_targets = len(reference_ids)
    num_layers = len(backbone.layers)
    if num_layers < 1:
        raise ValueError("The base model must have at least one decoder layer.")
    exit_layers = torch.zeros(num_targets, dtype=torch.long, device=device)
    early_predictions = torch.empty_like(reference_ids)
    gru_state = None

    # 2. Token positions, without a KV cache.
    cache_position = torch.arange(seq_len, device=device)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)

    # 3. The same causal and left-padding attention mask as training.
    mask_value = torch.finfo(hidden_states.dtype).min
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=mask_value,
        dtype=hidden_states.dtype, device=device,
    ).triu(diagonal=1)
    causal_mask = causal_mask[None, None].expand(batch_size, 1, seq_len, seq_len)
    causal_mask = causal_mask.masked_fill(attention_mask[:, None, None, :].eq(0), mask_value)

    # 4. Prepare rotary embeddings, when required by this Transformers version.
    position_embeddings = None
    if hasattr(backbone, "rotary_emb"):
        position_embeddings = backbone.rotary_emb(hidden_states, position_ids)

    # 5. Run decoder layers sequentially and commit each token's first exit.
    for layer_index, layer in enumerate(backbone.layers):
        parameters = inspect.signature(layer.forward).parameters
        kwargs = {"attention_mask": causal_mask, "position_ids": position_ids}
        if "cache_position" in parameters:
            kwargs["cache_position"] = cache_position
        if "position_embeddings" in parameters:
            kwargs["position_embeddings"] = position_embeddings
        if "use_cache" in parameters:
            kwargs["use_cache"] = False
        if "output_attentions" in parameters:
            kwargs["output_attentions"] = False
        layer_output = layer(hidden_states, **kwargs)
        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        raw = hidden_states[batch_indices, source_positions]  # [total targets, H]

        if layer_index == num_layers - 1:
            # Every undecided token falls back to the final layer. Its prediction
            # is also the baseline, including for tokens which already exited.
            final_predictions = predict_tokens(model, raw, projection_chunk_size)
            undecided = exit_layers.eq(0)
            early_predictions[undecided] = final_predictions[undecided]
            exit_layers[undecided] = num_layers
        else:
            # One GRU sequence step per decoder layer, with one batch item per
            # target token. Resetting gru_state here would change the classifier.
            states, gru_state = classifier.gru(
                raw.to(dtype=classifier_parameter.dtype).unsqueeze(1), gru_state,
            )
            scores = classifier.head(states[:, 0]).squeeze(-1).sigmoid()
            if not torch.isfinite(scores).all().item():
                raise ValueError("The early-exit classifier produced non-finite scores.")
            exit_now = exit_layers.eq(0) & scores.ge(threshold)
            if exit_now.any().item():
                early_predictions[exit_now] = predict_tokens(
                    model, raw[exit_now], projection_chunk_size,
                )
                exit_layers[exit_now] = layer_index + 1

    rows = []
    for batch_index in range(batch_size):
        selected = batch_indices.eq(batch_index)
        rows.append({
            "reference_target_ids": reference_ids[selected].cpu(),
            "early_exit_prediction_ids": early_predictions[selected].cpu(),
            "final_prediction_ids": final_predictions[selected].cpu(),
            "exit_layers": exit_layers[selected].cpu(),
        })
    return rows


@dataclass
class AccuracyMetrics:
    num_layers: int
    documents: int = 0
    tokens: int = 0
    early_exit_correct: int = 0
    final_layer_correct: int = 0
    matches_final: int = 0
    early_exit_tokens: int = 0
    exit_layer_sum: int = 0
    exit_layer_counts: dict[int, int] = field(default_factory=dict)

    def update(self, rows):
        for row in rows:
            targets = row["reference_target_ids"]
            early = row["early_exit_prediction_ids"]
            final = row["final_prediction_ids"]
            layers = row["exit_layers"]
            self.documents += 1
            self.tokens += targets.numel()
            self.early_exit_correct += early.eq(targets).sum().item()
            self.final_layer_correct += final.eq(targets).sum().item()
            self.matches_final += early.eq(final).sum().item()
            self.early_exit_tokens += layers.lt(self.num_layers).sum().item()
            self.exit_layer_sum += layers.sum().item()
            values, counts = layers.unique(return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                self.exit_layer_counts[value] = self.exit_layer_counts.get(value, 0) + count

    def compute(self):
        if not self.tokens:
            raise ValueError("The selected dataset contains no summary tokens to evaluate.")
        return {
            "documents": self.documents,
            "tokens_including_eos": self.tokens,
            "early_exit_correct": self.early_exit_correct,
            "early_exit_accuracy": self.early_exit_correct / self.tokens,
            "final_layer_correct": self.final_layer_correct,
            "final_layer_accuracy": self.final_layer_correct / self.tokens,
            "agreement_with_final_layer": self.matches_final / self.tokens,
            "early_exit_tokens": self.early_exit_tokens,
            "early_exit_rate": self.early_exit_tokens / self.tokens,
            "mean_exit_layer": self.exit_layer_sum / self.tokens,
            "num_decoder_layers": self.num_layers,
            "exit_layer_counts": {
                str(layer): self.exit_layer_counts.get(layer, 0)
                for layer in range(1, self.num_layers + 1)
            },
        }


def evaluate(base_model, tokenizer, classifier, args):
    """Stream the selected test rows in dataset order, just as in training."""
    metrics = AccuracyMetrics(num_layers=len(base_model.model.layers))
    encoder = TokenEncoder(tokenizer, max_length=args.max_length)
    next_index = args.start_row
    current_index = next_index
    batch_indices = []
    try:
        records = iter(load_records(args))
        while True:
            examples, batch_indices = [], []
            for offset in range(args.batch_size):
                current_index = next_index + offset
                try:
                    example = next(records)
                except StopIteration:
                    break
                examples.append(example)
                batch_indices.append(current_index)
            if not examples:
                break
            current_index = batch_indices[0]
            batch = encoder(examples)
            rows = forward_layer_by_layer(
                base_model, classifier, **batch, threshold=args.threshold,
                projection_chunk_size=args.projection_chunk_size,
            )
            metrics.update(rows)
            next_index = batch_indices[-1] + 1
            result = metrics.compute()
            print(
                f"Dataset indices {batch_indices} | Documents: {metrics.documents} | "
                f"Tokens: {metrics.tokens} | Early-exit accuracy: "
                f"{result['early_exit_accuracy']:.4%} | Final-layer accuracy: "
                f"{result['final_layer_accuracy']:.4%} | "
                f"Mean exit layer: {result['mean_exit_layer']:.2f}/{metrics.num_layers} | "
                f"Exit layer counts (1-based): {result['exit_layer_counts']}", flush=True,
            )
            del examples, batch, rows
        return metrics.compute()
    except (Exception, KeyboardInterrupt):
        print(
            f"Evaluation stopped. Dataset: {args.dataset}, split: {args.split}. "
            f"Current dataset index (zero-based): {current_index}; "
            f"batch indices: {batch_indices}; next unprocessed dataset index: {next_index}.",
            file=sys.stderr, flush=True,
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ee_cnn/final.pt"))
    parser.add_argument("--dataset", default=None, help="Default: training checkpoint dataset.")
    parser.add_argument("--version", choices=VERSIONS, default=None)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=1000, help="Default: the entire test split.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--model-path", default=None, help="Default: the base model used for training.")
    parser.add_argument("--projection-chunk-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Exit at the first sigmoid score >= this value.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training_args = checkpoint["training_args"]
    defaults = {
        "dataset": DATASET, "version": "v3", "batch_size": 8,
        "max_length": 4096, "model_path": MODEL, "projection_chunk_size": 32,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, training_args.get(name, fallback))
    for name in ("batch_size", "max_length", "projection_chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.start_row < 0 or (args.max_documents is not None and args.max_documents < 1):
        parser.error("--start-row must be non-negative and --max-documents must be positive.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    device = torch.device(args.device)
    classifier = EarlyExitGRU(**checkpoint["model_config"])
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier = classifier.to(device).eval().requires_grad_(False)
    del checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    if base_model.config.hidden_size != classifier.config["input_size"]:
        raise ValueError("The base model hidden size does not match the classifier checkpoint.")
    args.max_length = min(args.max_length, base_model.config.max_position_embeddings)
    print(
        f"Checkpoint: {args.checkpoint} | Split: {args.split} | Threshold: {args.threshold}\n"
        "Teacher-forced summary tokens + EOS. All Llama layers run for attention "
        "context and the final-layer baseline; exit depth is not a speed measurement.",
        flush=True,
    )
    metrics = evaluate(base_model, tokenizer, classifier, args)
    report = {
        "evaluation_mode": "teacher_forced",
        "computes_all_decoder_layers": True,
        "settings": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        **metrics,
    }
    print(
        f"\nEvaluated {metrics['documents']} documents, {metrics['tokens_including_eos']} tokens including EOS.\n"
        f"Early-exit next-token accuracy: {metrics['early_exit_accuracy']:.4%} "
        f"({metrics['early_exit_correct']}/{metrics['tokens_including_eos']})\n"
        f"Final-layer next-token accuracy: {metrics['final_layer_accuracy']:.4%} "
        f"({metrics['final_layer_correct']}/{metrics['tokens_including_eos']})\n"
        f"Agreement with final layer: {metrics['agreement_with_final_layer']:.4%}\n"
        f"Early-exit rate: {metrics['early_exit_rate']:.4%}\n"
        f"Mean exit layer: {metrics['mean_exit_layer']:.2f}/{metrics['num_decoder_layers']}\n"
        f"Exit layer counts (1-based): {metrics['exit_layer_counts']}",
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved metrics: {args.output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    main()
