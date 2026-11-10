"""Train an early-exit GRU directly on CNN/DailyMail, without Parquet files.

Run: python train_ee_cnn.py --max-documents 100 --output-dir checkpoints/ee_cnn
Resume: python train_ee_cnn.py --resume checkpoints/ee_cnn/run_<id>/latest.pt
Tests: python -m pytest tests/test_train_ee_cnn.py

The frozen Llama supplies raw hidden_states [L, T, H] and boolean
matches_reference_or_final [L, T]. T counts summary tokens plus EOS.
Each token is an independent GRU sequence across decoder layers.

Checkpoints contain GRU weights, optimizer state, model settings, RNG state,
and the next unprocessed dataset row. Dataset indices and epochs in checkpoint
metadata are zero-based. Training preserves dataset order. Downloads and
training occur only when main() is run.

latest.pt is saved every --save-every updates and at each epoch boundary.
Finished epochs also save epoch_XXXX.pt; the final model is saved in final.pt.
Each invocation creates a unique run directory under --output-dir, including
resumed runs. best.pt keeps the update with the lowest batch training loss in
this run; completion copies it to ee_gru.pt. A resumed checkpoint with a known
loss initializes the new run's best checkpoint. Loss is measured before each
update; checkpoints contain the resulting updated weights.
Per-batch early_exit_next_token_accuracy averages reference-token correctness
over summary tokens plus EOS, choosing the first sigmoid score >= 0.5 and
falling back to the final layer. Padding is excluded.
Exceptions and Ctrl+C save exception.pt and report the current dataset indices
before re-raising. Resume restores the original data/model settings; --epochs
sets the total number of epochs, including those already completed.
"""

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import inspect
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET = "abisee/cnn_dailymail"
MODEL = "facebook/layerskip-llama2-7B"
VERSIONS = {"v1": "1.0.0", "v2": "2.0.0", "v3": "3.0.0"}
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def load_records(args, start_row=None):
    """Return a reusable streaming dataset, starting at a zero-based row."""
    if args.start_row < 0:
        raise ValueError("start_row must be non-negative.")
    if args.max_documents is not None and args.max_documents < 1:
        raise ValueError("max_documents must be positive or None.")
    start_row = args.start_row if start_row is None else start_row
    if start_row < args.start_row:
        raise ValueError("Cannot resume before the selected starting row.")

    dataset = load_dataset(
        args.dataset,
        name=VERSIONS[args.version],
        split=SPLITS[args.split],
        streaming=True,
    )
    if start_row:
        dataset = dataset.skip(start_row)
    if args.max_documents is not None:
        remaining = args.start_row + args.max_documents - start_row
        dataset = dataset.take(max(0, remaining))
    return dataset


class TokenEncoder:
    """Cache fixed tokens and batch-tokenize variable text once per batch."""

    def __init__(self, tokenizer, task="summarize", max_length=4096):
        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("This function requires BOS and EOS tokens.")
        if task != "summarize":
            raise ValueError(f"Unsupported task: {task}")
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_id

        self.prefix = [tokenizer.bos_token_id] + tokenizer.encode(
            "[Prompt] ", add_special_tokens=False
        )
        self.separator = tokenizer.encode(
            " [Summary] ", add_special_tokens=False
        )

    def encode(self, prompt, target):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        targets = [target] if isinstance(target, str) else list(target)
        if not prompts:
            raise ValueError("The batch must not be empty.")
        if len(prompts) != len(targets):
            raise ValueError("prompt and target must have the same batch size.")

        batch_size = len(prompts)
        # Tokenize segments separately to keep the summary boundary explicit.
        token_lists = self.tokenizer(
            prompts + targets,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        examples = []
        for i in range(batch_size):
            target_ids = token_lists[batch_size + i] + [self.eos_id]
            prompt_budget = (
                self.max_length - len(self.prefix)
                - len(self.separator) - len(target_ids)
            )
            if prompt_budget < 0:
                raise ValueError(
                    "The full summary, EOS, BOS, and markers cannot fit in max_length."
                )

            prompt_ids = (
                self.prefix + token_lists[i][:prompt_budget] + self.separator
            )
            examples.append((prompt_ids + target_ids, len(target_ids)))

        batch_length = max(len(ids) for ids, _ in examples)
        input_ids = torch.full(
            (batch_size, batch_length), self.pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)

        for i, (ids, target_length) in enumerate(examples):
            start = batch_length - len(ids)
            input_ids[i, start:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, start:] = 1

            # Standard causal-LM labels stay at the target's input position.
            # Shift source hidden-state positions, not labels, during extraction.
            target_start = batch_length - target_length
            labels[i, target_start:] = input_ids[i, target_start:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, examples):
        return self.encode(
            [example["article"] for example in examples],
            [example["highlights"] for example in examples],
        )


@torch.no_grad()
def forward_layer_by_layer(
    model, input_ids, attention_mask, labels, projection_chunk_size=32,
):
    """Return hidden states and training/accuracy labels per document on CPU.

    Keep raw decoder outputs on the residual path. For each layer, apply the
    final norm once to the selected states before the shared vocabulary head.
    The model must fit on one device (the command-line runner ensures this).
    """
    # no_grad creates ordinary tensors which the GRU can use for backward.
    if projection_chunk_size < 1:
        raise ValueError("projection_chunk_size must be positive.")
    model.eval()
    backbone = model.model
    device = backbone.embed_tokens.weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # 1. Token IDs -> embeddings.
    # Shape: [batch_size, sequence_length, hidden_size]
    hidden_states = backbone.embed_tokens(input_ids)

    batch_size, seq_len, _ = hidden_states.shape
    rows = []
    source_positions = []
    reference_targets = []
    prediction_ids = []
    num_layers = len(backbone.layers)
    for batch_index in range(batch_size):
        target_positions = labels[batch_index].ne(-100).nonzero().flatten()
        if not len(target_positions) or target_positions[0].item() == 0:
            raise ValueError("Each document needs targets with preceding input tokens.")
        # Input position j predicts j + 1: start just before t1 and end at tM.
        source_positions.append(target_positions - 1)
        target_count = len(target_positions)  # T = summary tokens M + EOS.
        reference_targets.append(labels[batch_index, target_positions].cpu())
        prediction_ids.append(torch.empty(num_layers, target_count, dtype=torch.long))
        rows.append({
            "hidden_states": torch.empty(
                num_layers, target_count, hidden_states.shape[-1], dtype=torch.float32
            ),
        })

    # 2. Token positions. This example does not use a KV cache.
    cache_position = torch.arange(seq_len, device=device)
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)

    # 3. Build the causal attention mask.
    # 0 = allowed; large negative value = blocked.
    mask_value = torch.finfo(hidden_states.dtype).min

    causal_mask = torch.full(
        (seq_len, seq_len),
        fill_value=mask_value,
        dtype=hidden_states.dtype,
        device=device,
    ).triu(diagonal=1)

    # Shape: [batch_size, 1, sequence_length, sequence_length]
    causal_mask = causal_mask[None, None].expand(
        batch_size, 1, seq_len, seq_len
    )

    # Also block attention to padding tokens.
    padding_mask = attention_mask[:, None, None, :] == 0
    causal_mask = causal_mask.masked_fill(padding_mask, mask_value)

    # 4. Prepare rotary position embeddings.
    # Older Transformers versions compute these inside attention instead.
    position_embeddings = None
    if hasattr(backbone, "rotary_emb"):
        position_embeddings = backbone.rotary_emb(
            hidden_states, position_ids
        )

    # 5. Run decoder layers sequentially.
    for layer_index, layer in enumerate(backbone.layers):
        # Handle differences between Transformers layer APIs.
        parameters = inspect.signature(layer.forward).parameters

        kwargs = {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
        }

        if "cache_position" in parameters:
            kwargs["cache_position"] = cache_position

        if "position_embeddings" in parameters:
            kwargs["position_embeddings"] = position_embeddings

        if "use_cache" in parameters:
            kwargs["use_cache"] = False

        if "output_attentions" in parameters:
            kwargs["output_attentions"] = False

        layer_output = layer(hidden_states, **kwargs)

        # Older versions return a tuple; newer versions return a tensor.
        hidden_states = (
            layer_output[0]
            if isinstance(layer_output, tuple)
            else layer_output
        )

        for batch_index, row in enumerate(rows):
            raw = hidden_states[batch_index, source_positions[batch_index]]  # [T, H]
            row["hidden_states"][layer_index].copy_(raw)
            # Do not feed this normalized tensor into the next decoder layer.
            # Argmax does not need softmax or stored vocabulary distributions.
            for start in range(0, len(raw), projection_chunk_size):
                stop = start + projection_chunk_size
                logits = model.lm_head(backbone.norm(raw[start:stop]))
                prediction_ids[batch_index][layer_index, start:stop].copy_(
                    logits.argmax(dim=-1)
                )
                del logits

    for row, predictions, targets in zip(rows, prediction_ids, reference_targets):
        row["matches_reference"] = predictions == targets[None, :]
        row["matches_reference_or_final"] = (
            row["matches_reference"]
            | (predictions == predictions[-1:])
        )
    return rows


class EarlyExitGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Each token is an independent sequence across LLM layers.
        batch_size, layers, tokens, hidden_size = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * tokens, layers, hidden_size)
        states, _ = self.gru(x)
        logits = self.head(states).squeeze(-1)
        return logits.reshape(batch_size, tokens, layers).transpose(1, 2)


def collate_batch(rows):
    """Pad only the token axis; -100 labels are excluded from the loss."""
    xs, ys = [], []
    for row in rows:
        x = row["hidden_states"]
        y = row["matches_reference_or_final"]
        if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
            raise ValueError("Expected aligned hidden_states [L, T, H] and labels [L, T].")
        if not all(x.shape):
            raise ValueError("Hidden states must have nonempty layer, token, and feature axes.")
        xs.append(x.transpose(0, 1))
        ys.append(y.float().transpose(0, 1))
    x = pad_sequence(xs, batch_first=True, padding_value=0)
    y = pad_sequence(ys, batch_first=True, padding_value=-100)
    return x.transpose(1, 2), y.transpose(1, 2)


@torch.no_grad()
def early_exit_next_token_accuracy(logits, rows):
    """Token-weighted batch accuracy using the first accepted layer (threshold 0.5)."""
    correct = total = 0
    for scores, row in zip(logits.detach().cpu(), rows):
        matches = row["matches_reference"]
        token_count = matches.shape[1]
        eligible = scores[:, :token_count].ge(0)  # sigmoid(logit) >= 0.5
        eligible[-1] = True
        exits = eligible.long().argmax(dim=0)
        correct += matches[exits, torch.arange(token_count)].sum().item()
        total += token_count
    return correct / total


@dataclass
class TrainingProgress:
    # epoch, batch_index, and next_dataset_index describe the next update.
    epoch: int = 0
    batch_index: int = 0
    global_step: int = 0
    next_dataset_index: int = 0
    current_dataset_index: int | None = None
    current_dataset_indices: list[int] = field(default_factory=list)
    last_completed_dataset_index: int | None = None
    last_loss: float | None = None
    best_loss: float | None = None
    stage: str = "initializing"


def cpu_state(value):
    """Keep checkpoints portable and avoid GPU allocations while saving."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_checkpoint(path, classifier, optimizer, progress, args, error=None):
    """Atomically replace a checkpoint, preserving the previous file on failure."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "checkpoint_path": str(path),
        "model_state_dict": cpu_state(classifier.state_dict()),
        "optimizer_state_dict": cpu_state(optimizer.state_dict()),
        "model_config": classifier.config,
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "progress": asdict(progress),
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved checkpoint: {path} | next dataset index: {progress.next_dataset_index}", flush=True)
    return path


def train(base_model, tokenizer, classifier, optimizer, args, progress=None):
    """Extract a batch, update the GRU, then commit its dataset cursor."""
    if progress is None:
        progress = TrainingProgress(
            next_dataset_index=args.start_row, current_dataset_index=args.start_row,
        )
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_prefix = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_")
    output_dir = Path(tempfile.mkdtemp(prefix=run_prefix, dir=output_root))
    args.output_dir = output_dir
    print(f"Run output directory: {output_dir}", flush=True)
    # Best loss belongs to this invocation; a resumed model is its baseline.
    progress.best_loss = progress.last_loss if args.resume is not None else None
    device = next(classifier.parameters()).device
    try:
        encoder = TokenEncoder(tokenizer, max_length=args.max_length)
        base_model.eval().requires_grad_(False)
        classifier.train()
        criterion = nn.BCEWithLogitsLoss()
        if progress.best_loss is not None:
            save_checkpoint(output_dir / "best.pt", classifier, optimizer, progress, args)
        if args.resume is None:
            save_checkpoint(output_dir / "latest.pt", classifier, optimizer, progress, args)

        while progress.epoch < args.epochs:
            progress.stage = "loading_dataset"
            progress.current_dataset_index = progress.next_dataset_index
            progress.current_dataset_indices = []
            records = iter(load_records(args, start_row=progress.next_dataset_index))
            while True:
                examples = []
                progress.current_dataset_indices = []
                progress.stage = "reading_dataset"
                for offset in range(args.batch_size):
                    index = progress.next_dataset_index + offset
                    progress.current_dataset_index = index
                    try:
                        example = next(records)
                    except StopIteration:
                        break
                    examples.append(example)
                    progress.current_dataset_indices.append(index)
                if not examples:
                    if progress.batch_index == 0:
                        raise ValueError("The selected dataset contains no documents.")
                    break

                progress.current_dataset_index = progress.current_dataset_indices[0]
                progress.stage = "encoding"
                batch = encoder(examples)
                progress.stage = "extracting_hidden_states"
                rows = forward_layer_by_layer(
                    base_model, **batch, projection_chunk_size=args.projection_chunk_size,
                )
                x, y = collate_batch(rows)
                del batch, examples
                progress.stage = "training"
                x, y = x.to(device), y.to(device)
                valid_mask = y.ne(-100)
                optimizer.zero_grad(set_to_none=True)
                logits = classifier(x)
                loss = criterion(logits[valid_mask], y[valid_mask])
                if not torch.isfinite(loss).item():
                    raise ValueError("Training loss is not finite.")
                loss_value = loss.item()
                batch_accuracy = early_exit_next_token_accuracy(logits, rows)
                del rows
                loss.backward()
                progress.stage = "optimizer_step"
                optimizer.step()
                # Advance only after a successful update. Failed batches can be retried.
                progress.next_dataset_index = progress.current_dataset_indices[-1] + 1
                progress.last_completed_dataset_index = progress.current_dataset_indices[-1]
                progress.global_step += 1
                progress.batch_index += 1
                progress.last_loss = loss_value
                progress.stage = "batch_completed"
                if progress.best_loss is None or loss_value < progress.best_loss:
                    progress.best_loss = loss_value
                    save_checkpoint(output_dir / "best.pt", classifier, optimizer, progress, args)
                del x, y, valid_mask, logits, loss
                print(
                    f"Epoch {progress.epoch + 1}/{args.epochs} | "
                    f"Batch {progress.batch_index} | Dataset indices "
                    f"{progress.current_dataset_indices} | Loss: {loss_value:.6f} | "
                    f"early_exit_next_token_accuracy: {batch_accuracy:.4%}",
                    flush=True,
                )
                if progress.global_step % args.save_every == 0:
                    save_checkpoint(output_dir / "latest.pt", classifier, optimizer, progress, args)

            progress.epoch += 1
            progress.batch_index = 0
            progress.next_dataset_index = args.start_row
            progress.current_dataset_index = None
            progress.current_dataset_indices = []
            progress.stage = "epoch_completed"
            save_checkpoint(
                output_dir / f"epoch_{progress.epoch:04d}.pt",
                classifier, optimizer, progress, args,
            )
            save_checkpoint(output_dir / "latest.pt", classifier, optimizer, progress, args)

        progress.stage = "completed"
        final_path = save_checkpoint(output_dir / "final.pt", classifier, optimizer, progress, args)
        if (output_dir / "best.pt").exists():
            shutil.copyfile(output_dir / "best.pt", output_dir / "ee_gru.pt")
            print(f"Copied best checkpoint to: {output_dir / 'ee_gru.pt'}", flush=True)
        return final_path
    except (Exception, KeyboardInterrupt) as error:
        print(
            f"Training stopped during {progress.stage}: {type(error).__name__}: {error}\n"
            f"Dataset: {args.dataset}, split: {args.split}. "
            f"Current dataset index (zero-based): {progress.current_dataset_index}; "
            f"batch indices: {progress.current_dataset_indices}. "
            f"Next unprocessed dataset index: {progress.next_dataset_index}.",
            file=sys.stderr, flush=True,
        )
        try:
            save_checkpoint(
                output_dir / "exception.pt", classifier, optimizer, progress, args, error=error,
            )
        except Exception as checkpoint_error:
            print(f"Could not save exception checkpoint: {checkpoint_error}", file=sys.stderr, flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--max-documents", type=int, default=None, help="Default: the entire selected split.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--version", choices=VERSIONS, default="v3")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--model-path", default=MODEL, help="Frozen Llama model ID or directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs, including resumed epochs (default: 3).")
    parser.add_argument("--gru-hidden-size", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--projection-chunk-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100, help="Save latest.pt every N optimizer updates.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Parent for a unique run directory (default: checkpoints/ee_cnn).")
    parser.add_argument(
        "--resume", type=Path, help="Restore weights, optimizer, cursor, and original data/model settings.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    checkpoint = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["error"] and checkpoint["progress"]["stage"] == "optimizer_step":
            parser.error(
                "This checkpoint may contain a partial optimizer update. Resume latest.pt "
                "or an epoch checkpoint to use a fully completed update."
            )
        for name in (
            "dataset", "version", "split", "start_row", "max_documents", "batch_size",
            "max_length", "model_path", "gru_hidden_size", "gru_layers", "lr",
            "projection_chunk_size", "seed",
        ):
            setattr(args, name, checkpoint["training_args"][name])
    if args.epochs is None:
        args.epochs = checkpoint["training_args"]["epochs"] if checkpoint else 3
    if args.output_dir is None:
        args.output_dir = Path("checkpoints/ee_cnn")
    for name in ("batch_size", "max_length", "epochs", "gru_hidden_size", "gru_layers", "projection_chunk_size", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.start_row < 0 or (args.max_documents is not None and args.max_documents < 1):
        parser.error("--start-row must be non-negative and --max-documents must be positive.")
    if not 0 < args.lr < float("inf"):
        parser.error("--lr must be finite and positive.")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    args.max_length = min(args.max_length, base_model.config.max_position_embeddings)
    classifier = EarlyExitGRU(
        input_size=base_model.config.hidden_size,
        hidden_size=args.gru_hidden_size,
        num_layers=args.gru_layers,
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    progress = None
    if checkpoint is not None:
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        progress = TrainingProgress(**checkpoint["progress"])
        torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(
            f"Resuming {args.resume} | epoch {progress.epoch + 1} | "
            f"next dataset index: {progress.next_dataset_index}", flush=True,
        )
        del checkpoint
    return train(base_model, tokenizer, classifier, optimizer, args, progress)


if __name__ == "__main__":
    main()
