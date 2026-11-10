"""Extract all decoder layers for three CNN/DailyMail reference summaries.

Run: python generate_cnn_dataset.py --output cnn_layer_outputs.parquet
Tests: python -m pytest tests/test_generate_cnn_dataset.py

Each Parquet row contains one unpadded document. Nested Arrow lists preserve
tensor axes: IDs/labels/masks are int64, flags are bool, and raw hidden states
and full-vocabulary probabilities are float32. Use reconstruct_row() to read
them as CPU tensors, without converting large arrays to Python float lists.

This is teacher-forced inference. Downloads occur only when main() is run.
"""

import argparse
import inspect
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET = "abisee/cnn_dailymail"
MODEL = "facebook/layerskip-llama2-7B"
NUM_DOCUMENTS = 3
VERSIONS = {"v1": "1.0.0", "v2": "2.0.0", "v3": "3.0.0"}
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def load_records(args):
    """Return a reusable streaming dataset, starting at a zero-based row."""
    if args.start_row < 0:
        raise ValueError("start_row must be non-negative.")
    if args.max_documents is not None and args.max_documents < 0:
        raise ValueError("max_documents must be non-negative or None.")

    dataset = load_dataset(
        args.dataset,
        name=VERSIONS[args.version],
        split=SPLITS[args.split],
        streaming=True,
    )
    if args.start_row:
        dataset = dataset.skip(args.start_row)
    if args.max_documents is not None:
        dataset = dataset.take(args.max_documents)
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

    # TODO: Modify if using different dataset 
    def __call__(self, examples):
        return self.encode(
            [example["article"] for example in examples],
            [example["highlights"] for example in examples],
        )


def encode_and_collate(tokenizer, prompt, target, task="summarize", max_length=4096):
    """Compatibility wrapper; reuse TokenEncoder for repeated calls."""
    return TokenEncoder(tokenizer, task, max_length).encode(prompt, target)


def make_dataloader(tokenizer, args):
    return DataLoader(
        load_records(args),
        batch_size=args.batch_size,
        collate_fn=TokenEncoder(tokenizer, max_length=args.max_length),
        num_workers=0,  # Preserve exact stream order and skip/take semantics.
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def forward_layer_by_layer(model, input_ids, attention_mask, labels):
    """Return one CPU tensor dictionary per document, selecting only T targets.

    Keep raw decoder outputs on the residual path. For each layer, apply the
    final norm once to the selected states before the shared vocabulary head.
    The model must fit on one device (the command-line runner ensures this).
    """
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
    num_layers = len(backbone.layers)
    for batch_index in range(batch_size):
        valid = attention_mask[batch_index].bool()
        target_positions = labels[batch_index].ne(-100).nonzero().flatten()
        if not len(target_positions) or target_positions[0].item() == 0:
            raise ValueError("Each document needs targets with preceding input tokens.")
        # Input position j predicts j + 1: start just before t1 and end at tM.
        source_positions.append(target_positions - 1)
        target_count = len(target_positions)  # T = summary tokens M + EOS.
        rows.append({
            "input_ids": input_ids[batch_index, valid].cpu(),
            "labels": labels[batch_index, valid].cpu(),
            "attention_mask": attention_mask[batch_index, valid].cpu(),
            "reference_target_ids": labels[batch_index, target_positions].cpu(),
            "hidden_outputs": torch.empty(
                num_layers, target_count, hidden_states.shape[-1], dtype=torch.float32
            ),
            "prediction_ids": torch.empty(num_layers, target_count, dtype=torch.long),
            "distributions": torch.empty(
                num_layers, target_count, model.config.vocab_size, dtype=torch.float32
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
            row["hidden_outputs"][layer_index].copy_(raw)
            # Do not feed this normalized tensor into the next decoder layer.
            logits = model.lm_head(backbone.norm(raw))  # [T, V]
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            row["distributions"][layer_index].copy_(probabilities)
            row["prediction_ids"][layer_index].copy_(probabilities.argmax(dim=-1))

    for row in rows:
        predictions = row["prediction_ids"]  # [L, T]
        row["matches_reference_or_final"] = (
            (predictions == row["reference_target_ids"][None, :])
            | (predictions == predictions[-1:])
        )
    return rows


PARQUET_SCHEMA = pa.schema([
    pa.field("input_ids", pa.large_list(pa.int64()), metadata={b"shape": b"[S]"}),
    pa.field("labels", pa.large_list(pa.int64()), metadata={b"shape": b"[S]"}),
    pa.field("attention_mask", pa.large_list(pa.int64()), metadata={b"shape": b"[S]"}),
    pa.field("reference_target_ids", pa.large_list(pa.int64()), metadata={b"shape": b"[T]"}),
    pa.field("hidden_outputs", pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
             metadata={b"shape": b"[L, T, H]"}),
    pa.field("prediction_ids", pa.large_list(pa.large_list(pa.int64())),
             metadata={b"shape": b"[L, T]"}),
    pa.field("matches_reference_or_final", pa.large_list(pa.large_list(pa.bool_())),
             metadata={b"shape": b"[L, T]"}),
    pa.field("distributions", pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
             metadata={b"shape": b"[L, T, V]"}),
], metadata={
    b"alignment": b"T = M + 1; raw hidden at j predicts label at j + 1; EOS excluded as source",
    b"storage": b"hidden_outputs and distributions: float32; IDs/labels/masks: int64; flags: bool",
    b"projection": b"softmax(lm_head(final_norm(raw_decoder_output))), one norm per state",
})


def tensor_column(tensor):
    """Wrap a contiguous CPU tensor in nested Arrow lists containing one row."""
    values = pa.array(tensor.contiguous().numpy().reshape(-1))
    for size in reversed(tensor.shape):
        # Full-vocabulary tensors can exceed the 32-bit list-offset limit.
        offsets = pa.array(range(0, len(values) + 1, size), type=pa.int64())
        values = pa.LargeListArray.from_arrays(offsets, values)
    return values


def write_parquet(model, dataloader, output_path):
    """Write one document per row group, keeping only one batch in host RAM."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    row_count = 0
    try:
        with pq.ParquetWriter(
            temporary_path, PARQUET_SCHEMA, compression="zstd", use_dictionary=False,
        ) as writer:
            for batch in dataloader:
                rows = forward_layer_by_layer(model, **batch)
                if row_count + len(rows) > NUM_DOCUMENTS:
                    raise ValueError(f"Expected exactly {NUM_DOCUMENTS} documents.")
                for row in rows:
                    table = pa.Table.from_arrays(
                        [tensor_column(row[name]) for name in PARQUET_SCHEMA.names],
                        schema=PARQUET_SCHEMA,
                    )
                    writer.write_table(table, row_group_size=1)
                    row_count += 1
                    del table
                del rows, row  # Release the batch before extracting the next one.
        if row_count != NUM_DOCUMENTS:
            raise ValueError(f"Expected exactly {NUM_DOCUMENTS} documents, got {row_count}.")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def reconstruct_row(table, row_index=0):
    """Rebuild all eight CPU tensors from a PyArrow Table or RecordBatch row."""
    tensors = {}
    for name in PARQUET_SCHEMA.names:
        values = table[name][row_index].values
        shape = [len(values)]
        while pa.types.is_large_list(values.type):
            lengths = values.value_lengths().to_numpy()
            if not len(lengths) or (lengths != lengths[0]).any():
                raise ValueError(f"Column {name} does not contain a rectangular tensor.")
            shape.append(int(lengths[0]))
            values = values.flatten()
        # Copy Arrow's immutable buffer so the resulting tensor is writable.
        tensors[name] = torch.from_numpy(values.to_numpy(zero_copy_only=False).copy()).reshape(shape)
    return tensors


def verify_parquet(output_path, eos_token_id):
    """Read back all three rows and check unshifted labels and T alignment."""
    with pq.ParquetFile(output_path) as parquet:
        assert parquet.metadata.num_rows == NUM_DOCUMENTS, "Expected exactly 3 rows."
        assert parquet.schema_arrow.names == PARQUET_SCHEMA.names
        print("Columns:", parquet.schema_arrow.names)
        for row_index, batch in enumerate(parquet.iter_batches(batch_size=1)):
            row = reconstruct_row(batch)
            del batch
            if row_index == 0:
                for name in row:
                    print(f"  {name}: {tuple(row[name].shape)} ({row[name].dtype})")

            input_ids, labels = row["input_ids"], row["labels"]
            targets = row["reference_target_ids"]
            target_positions = labels.ne(-100).nonzero().flatten()
            target_count = targets.numel()  # M summary tokens plus one EOS.
            assert labels.shape == row["attention_mask"].shape == input_ids.shape
            assert row["attention_mask"].eq(1).all(), "Padding was serialized."
            assert target_count >= 1 and target_positions.numel() == target_count
            assert target_positions[0] > 0
            assert torch.equal(target_positions, torch.arange(len(input_ids) - target_count, len(input_ids)))
            assert torch.equal(labels[target_positions], input_ids[target_positions])
            assert torch.equal(targets, input_ids[target_positions])
            assert targets[-1].item() == eos_token_id
            assert (target_positions - 1)[-1] == len(input_ids) - 2  # tM, never EOS.

            predictions = row["prediction_ids"]
            flags = row["matches_reference_or_final"]
            assert predictions.ndim == 2 and predictions.shape[1] == target_count
            assert row["hidden_outputs"].ndim == row["distributions"].ndim == 3
            assert row["hidden_outputs"].shape[:2] == predictions.shape
            assert row["distributions"].shape[:2] == predictions.shape == flags.shape
            expected_flags = (predictions == targets[None, :]) | (predictions == predictions[-1:])
            assert torch.equal(flags, expected_flags) and flags[-1].all()
            # Check one layer at a time to bound temporary full-vocabulary arrays.
            for layer_index, probabilities in enumerate(row["distributions"]):
                assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(target_count))
                assert torch.equal(predictions[layer_index], probabilities.argmax(-1))
            del row, probabilities
    print(f"Verified {NUM_DOCUMENTS} documents with T = M + 1 aligned predictions each.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--max-documents", type=int, default=NUM_DOCUMENTS, choices=[NUM_DOCUMENTS],
        help="Number of documents (this export requires exactly 3).",
    )
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--version", choices=VERSIONS, default="v3")
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--output", type=Path, default=Path("cnn_layer_outputs.parquet"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive.")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()
    args.max_length = min(args.max_length, model.config.max_position_embeddings)
    write_parquet(model, make_dataloader(tokenizer, args), args.output)
    verify_parquet(args.output, tokenizer.eos_token_id)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
