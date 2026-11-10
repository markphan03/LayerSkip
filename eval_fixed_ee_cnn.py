"""Evaluate fixed-layer early exit on CNN/DailyMail next-token prediction.

Run: python eval_fixed_ee_cnn.py --output layer4_metrics.json
Tests: python -m pytest tests/test_eval_fixed_ee_cnn.py

Every prediction exits at decoder layer 4 by default (1-based numbering:
model.model.layers[3]). Later decoder layers are not executed. Predictions are
argmax(lm_head(final_norm(raw_layer_output))), just as in eval_ee_cnn.py.

The checkpoint supplies only the base-model and preprocessing settings. No
GRU weights or classifier scores are used. The test split, starting row, and
document limit are independent of the training selection. By default, evaluate
the first 1,000 test documents, matching eval_ee_cnn.py.

Accuracy is token-weighted exact match against reference-summary tokens plus
EOS, using teacher forcing. Article tokens and padding are excluded. JSON
accuracies are fractions in [0, 1]; terminal output displays percentages.
This does not generate autoregressive summaries or measure decoding speed.
"""

import argparse
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ee_cnn import predict_tokens
from train_ee_cnn import DATASET, MODEL, SPLITS, VERSIONS, TokenEncoder, load_records


@torch.inference_mode()
def forward_layer_by_layer(
    model, input_ids, attention_mask, labels, exit_layer=4, projection_chunk_size=32,
):
    """Stop at the requested decoder block and return per-document CPU IDs."""
    backbone = model.model
    num_layers = len(backbone.layers)
    if not 1 <= exit_layer <= num_layers:
        raise ValueError(f"exit_layer must be between 1 and {num_layers} (1-based).")
    if projection_chunk_size < 1:
        raise ValueError("projection_chunk_size must be positive.")
    model.eval()
    device = backbone.embed_tokens.weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # 1. The source state at j predicts the reference token at j + 1.
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

    # 2. Token positions, without a KV cache.
    cache_position = torch.arange(seq_len, device=device)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)

    # 3. The same causal and left-padding mask as training and eval_ee_cnn.py.
    mask_value = torch.finfo(hidden_states.dtype).min
    causal_mask = torch.full(
        (seq_len, seq_len), fill_value=mask_value,
        dtype=hidden_states.dtype, device=device,
    ).triu(diagonal=1)
    causal_mask = causal_mask[None, None].expand(batch_size, 1, seq_len, seq_len)
    causal_mask = causal_mask.masked_fill(attention_mask[:, None, None, :].eq(0), mask_value)

    # 4. Prepare rotary embeddings for the installed Transformers version.
    position_embeddings = None
    if hasattr(backbone, "rotary_emb"):
        position_embeddings = backbone.rotary_emb(hidden_states, position_ids)

    # 5. Run only the requested number of decoder blocks. Preserve raw states
    #    on the residual path and normalize once when projecting predictions.
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
        if layer_index + 1 == exit_layer:
            break

    raw = hidden_states[batch_indices, source_positions]
    predictions = predict_tokens(model, raw, projection_chunk_size)
    rows = []
    for batch_index in range(batch_size):
        selected = batch_indices.eq(batch_index)
        rows.append({
            "reference_target_ids": reference_ids[selected].cpu(),
            "early_exit_prediction_ids": predictions[selected].cpu(),
            "exit_layers": torch.full_like(reference_ids[selected], exit_layer).cpu(),
        })
    return rows


@dataclass
class AccuracyMetrics:
    documents: int = 0
    tokens: int = 0
    correct: int = 0

    def update(self, rows):
        for row in rows:
            targets = row["reference_target_ids"]
            predictions = row["early_exit_prediction_ids"]
            self.documents += 1
            self.tokens += targets.numel()
            self.correct += predictions.eq(targets).sum().item()

    def compute(self):
        if not self.tokens:
            raise ValueError("The selected dataset contains no summary tokens to evaluate.")
        return {
            "documents": self.documents,
            "tokens_including_eos": self.tokens,
            "early_exit_correct": self.correct,
            "early_exit_accuracy": self.correct / self.tokens,
        }


def evaluate(base_model, tokenizer, args):
    """Stream test batches and accumulate exact-match next-token accuracy."""
    metrics = AccuracyMetrics()
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
                base_model, **batch, exit_layer=args.exit_layer,
                projection_chunk_size=args.projection_chunk_size,
            )
            metrics.update(rows)
            next_index = batch_indices[-1] + 1
            result = metrics.compute()
            print(
                f"Dataset indices {batch_indices} | Documents: {metrics.documents} | "
                f"Tokens: {metrics.tokens} | Layer {args.exit_layer} accuracy: "
                f"{result['early_exit_accuracy']:.4%}", flush=True,
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
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/ee_cnn/final.pt"),
        help="Read base-model/preprocessing settings only; classifier weights are not used.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--version", choices=VERSIONS, default=None)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=1000, help="Default: the first 1,000 test documents.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--projection-chunk-size", type=int, default=None)
    parser.add_argument("--exit-layer", type=int, default=4, help="Fixed decoder exit layer, numbered from 1 (default: 4).")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    training_args = checkpoint["training_args"]
    del checkpoint
    defaults = {
        "dataset": DATASET, "version": "v3", "batch_size": 8,
        "max_length": 4096, "model_path": MODEL, "projection_chunk_size": 32,
    }
    for name, fallback in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, training_args.get(name, fallback))
    for name in ("batch_size", "max_length", "projection_chunk_size", "exit_layer"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.start_row < 0 or args.max_documents < 1:
        parser.error("--start-row must be non-negative and --max-documents must be positive.")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    num_layers = len(base_model.model.layers)
    if args.exit_layer > num_layers:
        parser.error(f"--exit-layer cannot exceed the model's {num_layers} decoder layers.")
    args.max_length = min(args.max_length, base_model.config.max_position_embeddings)
    print(
        f"Split: {args.split} | Fixed exit layer: {args.exit_layer}/{num_layers} (1-based)\n"
        "Teacher-forced summary tokens + EOS. The decoder stops at the selected layer.",
        flush=True,
    )
    metrics = evaluate(base_model, tokenizer, args)
    report = {
        "evaluation_mode": "teacher_forced",
        "exit_policy": "fixed_layer",
        "exit_layer": args.exit_layer,
        "num_decoder_layers": num_layers,
        "computes_all_decoder_layers": args.exit_layer == num_layers,
        "settings": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        **metrics,
    }
    print(
        f"\nEvaluated {metrics['documents']} documents, {metrics['tokens_including_eos']} tokens including EOS.\n"
        f"Layer {args.exit_layer} early-exit next-token accuracy: {metrics['early_exit_accuracy']:.4%} "
        f"({metrics['early_exit_correct']}/{metrics['tokens_including_eos']})",
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved metrics: {args.output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    main()
