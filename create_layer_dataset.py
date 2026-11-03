"""Stream CNN/DailyMail into context batches and per-token, per-layer Parquet.

Example (downloads and inference only occur when explicitly executed):
    python create_layer_dataset.py --model facebook/layerskip-llama2-7B \
        --output-dir layer_data --context-size 1024 --batch-size 2

Download just the test split, without loading a model:
    python create_layer_dataset.py --download-only --split test --output-dir cnn_data

CNN/DailyMail configurations (1.0.0, 2.0.0, 3.0.0) each contain train,
validation, and test. The default configuration is 3.0.0. Select others with
--configs 1.0.0 2.0.0 3.0.0; the earlier --versions v1 v2 v3 spelling also works.

Requires the repository dependencies and pyarrow (a datasets dependency).
Supported models expose model.layers, model.norm and an output embedding head,
as Llama/LayerSkip and Mistral do. Intermediate distributions use the logit lens:
apply the final normalization and LM head to each decoder layer's raw output.

Each feature row describes a prediction AFTER the input token at `position`.
`real_target` is the next input token, including across chunk boundaries, or
-100 at the end of a document. Padding produces no feature rows. Synthetic
summary repetitions are explicitly marked. This is teacher-forced extraction,
not autoregressive text generation.

Vector columns are little-endian binary arrays, avoiding Python float lists:
    np.frombuffer(row['distribution'], dtype='<f4')  # default probability dtype
    np.frombuffer(row['hidden_output'], dtype='<f2') # default hidden dtype
For float8 hidden outputs, reconstruct using
    torch.frombuffer(bytearray(blob), dtype=torch.uint8).view(torch.float8_e4m3fn)
Each split's manifest records dimensions, dtypes, label formulas, and shards.
"""

import argparse
import gc
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


IGNORE_INDEX = -100
VERSIONS = {"v1": "1.0.0", "v2": "2.0.0", "v3": "3.0.0"}
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def configuration_alias(value):
    for alias, config in VERSIONS.items():
        if value in (alias, config):
            return alias
    raise argparse.ArgumentTypeError("Expected 1.0.0, 2.0.0, 3.0.0, v1, v2, or v3")


def split_alias(value):
    return "val" if value == "validation" else value


def load_records(version, split, args):
    """Stream only the selected split from the selected dataset configuration."""
    dataset = load_dataset(
        args.dataset, name=VERSIONS[version], split=SPLITS[split],
        streaming=True, revision=args.dataset_revision,
    )
    records = iter(dataset)
    if args.max_documents is not None:
        records = itertools.islice(records, args.max_documents)
    return records


def available_memory():
    """Available host RAM, capped by Linux container limits when discoverable."""
    candidates = []
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        candidates.append(int(fields["MemAvailable"].split()[0]) * 1024)
    except (OSError, KeyError, ValueError):
        pass
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit = int(Path(limit_path).read_text())
            used = int(Path(used_path).read_text())
            candidates.append(max(0, limit - used))
        except (OSError, ValueError):
            pass
    return min(candidates) if candidates else 2**63 - 1


@dataclass
class Sequence:
    document_id: str
    sequence_id: int
    chunk_index: int
    token_ids: list
    real_targets: list
    is_summary: list
    is_synthetic: list
    length: int


def document_sequences(record, tokenizer, context_size):
    """Drop tails below 10%; repeat summary up to 90%; right-pad to context."""
    prefix = "[Prompt]\nSummarize the following document:\n" + record["article"]
    prefix += "\n\n[Answer]\nAnswer:\n"
    prompt = tokenizer.encode(prefix, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prompt.insert(0, tokenizer.bos_token_id)
    summary = tokenizer.encode(record["highlights"], add_special_tokens=False)
    tokens = prompt + summary + [tokenizer.eos_token_id]
    summary_flags = [False] * len(prompt) + [True] * (len(summary) + 1)
    fill_to = math.ceil(0.9 * context_size)
    for chunk_index, start in enumerate(range(0, len(tokens), context_size)):
        ids = tokens[start:start + context_size]
        flags = summary_flags[start:start + context_size]
        original_length = len(ids)
        if original_length < 0.1 * context_size:
            continue
        synthetic = [False] * original_length
        if original_length < fill_to:
            # Empty summaries use EOS as a defined fallback filler.
            filler = summary or [tokenizer.eos_token_id]
            count = fill_to - original_length
            ids += list(itertools.islice(itertools.cycle(filler), count))
            flags += [True] * count
            synthetic += [True] * count
        targets = ids[1:] + [IGNORE_INDEX]
        if original_length == context_size and start + context_size < len(tokens):
            targets[-1] = tokens[start + context_size]
        # EOS terminates the real document; artificial filler is never its target.
        if start + original_length == len(tokens):
            targets[original_length - 1] = IGNORE_INDEX
        length = len(ids)
        padding = context_size - length
        yield Sequence(
            document_id=str(record["id"]), sequence_id=-1, chunk_index=chunk_index,
            token_ids=ids + [tokenizer.pad_token_id] * padding,
            real_targets=targets + [IGNORE_INDEX] * padding,
            is_summary=flags + [False] * padding,
            is_synthetic=synthetic + [False] * padding, length=length,
        )


class ShardWriter:
    """Write bounded Arrow tables immediately; atomically publish closed shards."""

    def __init__(self, directory, prefix, max_bytes, reserve_bytes):
        self.directory = directory
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes
        self.writer = None
        self.rows = 0
        self.bytes = 0
        self.shards = []

    def write(self, table):
        if self.writer is not None and self.bytes + table.nbytes > self.max_bytes:
            self.close()
        if self.writer is None:
            self.path = self.directory / f"{self.prefix}_{len(self.shards) + 1:03d}.parquet"
            self.temporary = self.path.with_suffix(".parquet.partial")
            self.writer = pq.ParquetWriter(
                self.temporary, table.schema, compression="zstd", use_dictionary=False,
            )
        self.writer.write_table(table, row_group_size=table.num_rows)
        self.rows += table.num_rows
        self.bytes += table.nbytes
        if self.bytes >= self.max_bytes or available_memory() < self.reserve_bytes:
            self.close()

    def close(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        self.temporary.rename(self.path)
        self.shards.append({
            "file": self.path.name, "rows": self.rows,
            "uncompressed_bytes": self.bytes, "file_bytes": self.path.stat().st_size,
        })
        self.rows = self.bytes = 0


def binary_vectors(tensor, dtype):
    if dtype == "float8_e4m3fn":
        maximum = torch.finfo(torch.float8_e4m3fn).max
        array = tensor.float().clamp(-maximum, maximum).to(torch.float8_e4m3fn)
        array = array.contiguous().view(torch.uint8).cpu().numpy()
    else:
        array = tensor.float().cpu().numpy().astype(
            "<f2" if dtype == "float16" else "<f4", copy=False,
        )
    return pa.array([row.tobytes() for row in array], type=pa.binary())


def scalar_array(tensor, arrow_type):
    return pa.array(tensor.detach().cpu().numpy(), type=arrow_type)


def adaptive_alpha(confidence, args):
    if args.alpha_mode == "fixed":
        return torch.full_like(confidence, args.alpha)
    # Piecewise linear, monotone: p <= .2 -> .6; p = .9 -> .9; p = 1 -> 1.
    middle = 0.6 + (confidence - 0.2) * (0.3 / 0.7)
    return torch.where(confidence < 0.9, middle.clamp(min=0.6), confidence)


def capture_layers(model, sequences, args):
    """Offload raw decoder outputs immediately, avoiding all-layer GPU retention."""
    captured = [None] * len(model.model.layers)
    handles = []

    def hook(index):
        def save(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden.detach().to("cpu", copy=True)
        return save

    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([item.token_ids for item in sequences], device=device)
    attention_mask = torch.tensor(
        [[1] * item.length + [0] * (args.context_size - item.length) for item in sequences],
        device=device,
    )
    try:
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(hook(index)))
        output = model.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, output_hidden_states=False, return_dict=True,
        )
        final_hidden = output.last_hidden_state.detach().to("cpu", copy=True)
    finally:
        for handle in handles:
            handle.remove()
    return captured, final_hidden


def sequence_table(sequences, batch_id):
    return pa.table({
        "batch_id": pa.array([batch_id] * len(sequences), type=pa.int64()),
        "batch_slot": pa.array(range(len(sequences)), type=pa.int32()),
        "document_id": pa.array([s.document_id for s in sequences], type=pa.string()),
        "sequence_id": pa.array([s.sequence_id for s in sequences], type=pa.int64()),
        "chunk_index": pa.array([s.chunk_index for s in sequences], type=pa.int32()),
        "length": pa.array([s.length for s in sequences], type=pa.int32()),
        "token_ids": pa.array([s.token_ids for s in sequences], type=pa.list_(pa.int32())),
        "real_targets": pa.array([s.real_targets for s in sequences], type=pa.list_(pa.int32())),
        "attention_mask": pa.array(
            [[1] * s.length + [0] * (len(s.token_ids) - s.length) for s in sequences],
            type=pa.list_(pa.uint8()),
        ),
        "is_summary": pa.array([s.is_summary for s in sequences], type=pa.list_(pa.bool_())),
        "is_synthetic": pa.array([s.is_synthetic for s in sequences], type=pa.list_(pa.bool_())),
    })


@torch.inference_mode()
def extract_batch(model, sequences, batch_id, features, packed, args):
    captured, final_hidden = capture_layers(model, sequences, args)
    packed.write(sequence_table(sequences, batch_id))
    head = model.get_output_embeddings()
    head_device = head.weight.device
    norm = model.model.norm
    norm_device = next(norm.parameters()).device
    positions = [(b, p) for b, s in enumerate(sequences) for p in range(s.length)]
    vocab_size = head.weight.shape[0]
    hidden_size = head.weight.shape[1]
    # Bound projection workspace and serialized tables, independently of batch size.
    per_token = 40 * vocab_size + 8 * hidden_size
    budget = min(args.max_host_memory_gb * 2**30, available_memory() - args.reserve_bytes)
    shard_limit = args.max_shard_mb * 2**20
    stored_row_bytes = vocab_size * (2 if args.distribution_dtype == "float16" else 4)
    stored_row_bytes += hidden_size * {"float8_e4m3fn": 1, "float16": 2, "float32": 4}[args.hidden_dtype] + 512
    if budget < per_token:
        raise MemoryError("Insufficient host memory for one projected token; reduce batch/context size.")
    if shard_limit < stored_row_bytes:
        raise ValueError("--max-shard-mb cannot hold one feature row; increase it.")
    block_size = min(args.projection_tokens, int(budget // per_token), int(shard_limit // stored_row_bytes))
    for start in range(0, len(positions), block_size):
        block = positions[start:start + block_size]
        batch_indices = torch.tensor([b for b, _ in block])
        token_positions = torch.tensor([p for _, p in block])
        reference_hidden = final_hidden[batch_indices, token_positions].to(head_device)
        final_logp = head(reference_hidden).float().log_softmax(-1)
        final_probs = final_logp.exp()
        confidence, final_prediction = final_probs.max(-1)
        alpha = adaptive_alpha(confidence, args)
        already_matched = torch.zeros(len(block), dtype=torch.bool, device=head_device)
        common = {
            "batch_id": pa.array([batch_id] * len(block), type=pa.int64()),
            "batch_slot": pa.array([b for b, _ in block], type=pa.int32()),
            "sequence_id": pa.array([sequences[b].sequence_id for b, _ in block], type=pa.int64()),
            "position": pa.array([p for _, p in block], type=pa.int32()),
            "token_id": pa.array([sequences[b].token_ids[p] for b, p in block], type=pa.int32()),
            "real_target": pa.array([sequences[b].real_targets[p] for b, p in block], type=pa.int32()),
            "is_summary": pa.array([sequences[b].is_summary[p] for b, p in block], type=pa.bool_()),
            "is_synthetic": pa.array([sequences[b].is_synthetic[p] for b, p in block], type=pa.bool_()),
            "full_model_prediction": scalar_array(final_prediction, pa.int32()),
            "full_model_confidence": scalar_array(confidence, pa.float32()),
            "alpha": scalar_array(alpha, pa.float32()),
        }
        for layer_index, hidden in enumerate(captured):
            raw = hidden[batch_indices, token_positions]
            if layer_index == len(captured) - 1:
                logp = final_logp
            else:
                normalized = norm(raw.to(norm_device)).to(head_device)
                logp = head(normalized).float().log_softmax(-1)
            probabilities = logp.exp()
            top_probs, top_ids = probabilities.topk(min(10, vocab_size), dim=-1)
            prediction = probabilities.argmax(-1)
            matches = prediction.eq(final_prediction)
            first_match = matches & ~already_matched
            already_matched |= matches
            reference_probability = probabilities.gather(1, final_prediction[:, None]).squeeze(1)
            # Relative to THIS layer's argmax probability, using the final model's token.
            alpha_match = reference_probability >= alpha * probabilities.max(-1).values
            kl = (final_probs * (final_logp - logp)).sum(-1).clamp_min(0)
            columns = dict(common)
            columns.update({
                "layer": pa.array([layer_index + 1] * len(block), type=pa.int32()),
                "hidden_output": binary_vectors(raw, args.hidden_dtype),
                "distribution": binary_vectors(probabilities, args.distribution_dtype),
                "kl_to_final": scalar_array(kl, pa.float32()),
                "top10_token_ids": pa.array(top_ids.cpu().tolist(), type=pa.list_(pa.int32())),
                "top10_probabilities": pa.array(top_probs.cpu().tolist(), type=pa.list_(pa.float32())),
                "layer_prediction": scalar_array(prediction, pa.int32()),
                "final_token_probability": scalar_array(reference_probability, pa.float32()),
                "label_match": scalar_array(matches.to(torch.uint8), pa.uint8()),
                "label_first_match": scalar_array(first_match.to(torch.uint8), pa.uint8()),
                "label_alpha": scalar_array(alpha_match.to(torch.uint8), pa.uint8()),
            })
            features.write(pa.table(columns))
    del captured, final_hidden


def write_manifest(path, metadata):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def download_split(version, split, args):
    """Save original article/highlights/id rows, without tokenization or inference."""
    directory = args.output_dir / version / split
    directory.mkdir(parents=True, exist_ok=False)
    writer = ShardWriter(
        directory, f"{args.name}_{split}", args.max_shard_mb * 2**20, args.reserve_bytes,
    )
    metadata = {
        "status": "running", "mode": "download_only", "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_config": VERSIONS[version], "split": SPLITS[split], "documents": 0,
    }
    manifest_path = directory / "manifest.json"
    write_manifest(manifest_path, metadata)
    schema = pa.schema([(field, pa.string()) for field in ("id", "article", "highlights")])
    rows = []
    buffered_bytes = 0
    buffer_limit = min(8 * 2**20, args.max_shard_mb * 2**20, args.max_host_memory_gb * 2**28)

    def flush():
        if rows:
            writer.write(pa.Table.from_pylist(rows, schema=schema))
            metadata["documents"] += len(rows)
            rows.clear()

    try:
        for record in load_records(version, split, args):
            row = {field: record[field] for field in schema.names}
            row_bytes = sum(len(value.encode("utf-8")) for value in row.values())
            if rows and buffered_bytes + row_bytes > buffer_limit:
                flush()
                buffered_bytes = 0
            rows.append(row)
            buffered_bytes += row_bytes
            if len(rows) >= 256 or available_memory() < args.reserve_bytes:
                flush()
                buffered_bytes = 0
        flush()
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        writer.close()
        metadata["shards"] = writer.shards
        write_manifest(manifest_path, metadata)


def process_split(model, tokenizer, version, split, args):
    directory = args.output_dir / version / split
    # Never silently overwrite an earlier extraction, including partial runs.
    directory.mkdir(parents=True, exist_ok=False)
    features = ShardWriter(directory, f"{args.name}_{split}", args.max_shard_mb * 2**20, args.reserve_bytes)
    packed = ShardWriter(directory, f"{args.name}_sequences_{split}", args.max_shard_mb * 2**20, args.reserve_bytes)
    manifest_path = directory / "manifest.json"
    metadata = {
        "status": "running", "model": args.model, "model_revision": args.model_revision,
        "dataset": args.dataset, "dataset_revision": args.dataset_revision,
        "version": VERSIONS[version], "dataset_config": VERSIONS[version], "split": SPLITS[split],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "context_size": args.context_size, "requested_batch_size": args.batch_size,
        "last_batch": "keep smaller batch", "ignore_index": IGNORE_INDEX,
        "hidden_dtype": args.hidden_dtype, "distribution_dtype": args.distribution_dtype,
        "vector_encoding": "little-endian contiguous binary; float8 uses raw E4M3FN bytes, clipped to finite range",
        "hidden_size": model.config.hidden_size, "vocab_size": model.get_output_embeddings().weight.shape[0],
        "num_layers": len(model.model.layers), "layer_numbering": "1-based decoder layers; embedding excluded",
        "hidden_output": "raw decoder output before final normalization",
        "distribution": "softmax(lm_head(final_norm(raw_hidden)))",
        "kl_to_final": "KL(final || layer), natural logarithm; computed in float32",
        "label_match": "layer argmax equals final argmax",
        "label_first_match": "first matching layer per sequence position, including final layer",
        "label_alpha": "p_layer(final_argmax) >= alpha * max(p_layer)",
        "alpha_rule": "fixed --alpha or monotone interpolation (.2,.6), (.9,.9), (1,1), clipped below at .6",
        "special_tokens": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
        "documents": 0, "sequences": 0, "batches": 0,
    }
    write_manifest(manifest_path, metadata)
    batch = []
    sequence_id = 0
    # Raw hidden outputs plus normalized final states remain on CPU for one batch.
    bytes_per_sequence = args.context_size * model.config.hidden_size * model.dtype.itemsize
    bytes_per_sequence *= len(model.model.layers) + 1

    def batch_capacity():
        budget = min(args.max_host_memory_gb * 2**30, available_memory() - args.reserve_bytes)
        # Leave half of the extraction budget for Arrow and projection intermediates.
        capacity = min(args.batch_size, int(budget // (2 * bytes_per_sequence)))
        if capacity < 1:
            raise MemoryError("Host memory budget cannot hold one sequence; reduce --context-size.")
        return capacity

    def flush():
        extract_batch(model, batch, metadata["batches"], features, packed, args)
        metadata["batches"] += 1
        metadata["sequences"] += len(batch)
        batch.clear()

    try:
        capacity = batch_capacity()
        for record in load_records(version, split, args):
            metadata["documents"] += 1
            for sequence in document_sequences(record, tokenizer, args.context_size):
                sequence.sequence_id = sequence_id
                sequence_id += 1
                batch.append(sequence)
                if len(batch) == capacity:
                    flush()
                    capacity = batch_capacity()
        if batch:
            flush()
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        features.close()
        packed.close()
        metadata["feature_shards"] = features.shards
        metadata["sequence_shards"] = packed.shards
        write_manifest(manifest_path, metadata)
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="Required for layer extraction; unused with --download-only")
    parser.add_argument("--download-only", action="store_true", help="Save original dataset rows to Parquet without loading a model")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset", default="abisee/cnn_dailymail")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--configs", "--versions", dest="versions", nargs="+", type=configuration_alias,
        default=["v3"], metavar="CONFIG",
        help="Dataset configurations: 1.0.0, 2.0.0, 3.0.0 (default: 3.0.0); v1/v2/v3 also accepted",
    )
    parser.add_argument(
        "--splits", "--split", dest="splits", nargs="+", type=split_alias,
        choices=SPLITS, default=list(SPLITS),
        help="Splits to process (default: all); validation is an alias for val",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="cnn_dm")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--projection-tokens", type=int, default=16)
    parser.add_argument("--max-host-memory-gb", type=float, default=4)
    parser.add_argument("--memory-reserve-gb", type=float, default=1)
    parser.add_argument("--max-shard-mb", type=float, default=256)
    parser.add_argument("--max-documents", type=int, help="Optional limit per version/split")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--hidden-dtype", choices=["float8_e4m3fn", "float16", "float32"], default="float16")
    parser.add_argument("--distribution-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--alpha-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--alpha", type=float, default=0.8, help="Used with --alpha-mode fixed")
    args = parser.parse_args()
    if not args.download_only and not args.model:
        parser.error("--model is required unless --download-only is supplied")
    for key in ("context_size", "batch_size", "projection_tokens", "max_host_memory_gb", "max_shard_mb"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.context_size < 2:
        parser.error("--context-size must be at least 2")
    if not math.isfinite(args.memory_reserve_gb) or args.memory_reserve_gb < 0:
        parser.error("--memory-reserve-gb must be finite and nonnegative")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be positive")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must lie in [0, 1]")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a filename prefix without directories")
    if len(set(args.versions)) != len(args.versions) or len(set(args.splits)) != len(args.splits):
        parser.error("--configs/--versions and --splits must not contain duplicates")
    args.reserve_bytes = int(args.memory_reserve_gb * 2**30)
    return args


def main():
    args = parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("This writer requires a little-endian host.")
    for version in args.versions:
        for split in args.splits:
            if (args.output_dir / version / split).exists():
                raise FileExistsError(f"Output already exists: {args.output_dir / version / split}")
    if args.download_only:
        for version in args.versions:
            for split in args.splits:
                download_split(version, split, args)
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = "auto" if args.model_dtype == "auto" else getattr(torch, args.model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype=dtype,
        device_map="auto" if args.device == "auto" else {"": args.device},
    ).eval()
    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers") or not hasattr(backbone, "norm"):
        raise ValueError("Expected a Llama-compatible decoder with model.layers and model.norm.")
    # Disk/meta offload requires additional Accelerate hook coordination for direct projections.
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise ValueError("Disk/meta offloading is unsupported; use a model that fits CPU/GPU memory.")
    maximum_context = getattr(model.config, "max_position_embeddings", None)
    if maximum_context is not None and args.context_size > maximum_context:
        raise ValueError(f"--context-size exceeds model max_position_embeddings={maximum_context}")
    for version in args.versions:
        for split in args.splits:
            process_split(model, tokenizer, version, split, args)


if __name__ == "__main__":
    main()
