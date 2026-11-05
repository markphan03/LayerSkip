"""Collect labeled per-layer predictions during full-model greedy generation.

python create_layer_dataset_v2.py --model facebook/layerskip-llama2-7B \
    --output-dir layer_data --max-documents 1 --max-new-tokens 32

Each generation step records all layers on the same prompt + generated prefix.
Every prediction matching the final layer receives label_match=1; others get 0.
Reference summaries are never supplied. A KV cache reuses the prompt and earlier
tokens during decoding. Parquet and CSV are written.
"""

import argparse
import csv
import gc
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def document_prompt(record, tokenizer, context_size, max_new_tokens):
    """Truncate article tokens only; preserve BOS, prompt framing and answer cue."""
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    prefix = bos + tokenizer.encode("Article: ", add_special_tokens=False)
    suffix = tokenizer.encode("\nSummary:", add_special_tokens=False)
    article = tokenizer.encode(record["article"], add_special_tokens=False)
    article_budget = context_size - max_new_tokens - len(prefix) - len(suffix)
    if article_budget < 1:
        raise ValueError("Context size must fit prompt framing, article and --max-new-tokens.")
    truncated = max(0, len(article) - article_budget)
    if truncated:
        print(f"Document {record['id']}: truncating {truncated} article tokens.", flush=True)
    return prefix + article[:article_budget] + suffix, truncated


def export_parquet_csv(path):
    """Write a spreadsheet-readable copy, streaming one record batch at a time."""
    destination = path.with_suffix(".csv")
    temporary = destination.with_suffix(".csv.partial")
    parquet = pq.ParquetFile(path)
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(parquet.schema_arrow.names)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                writer.writerow([
                    json.dumps(value) if isinstance(value, (list, dict)) else value
                    for value in row.values()
                ])
    temporary.replace(destination)
    return destination


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
        csv_path = export_parquet_csv(self.path)
        self.shards.append({
            "file": self.path.name, "rows": self.rows,
            "csv_file": csv_path.name,
            "uncompressed_bytes": self.bytes, "file_bytes": self.path.stat().st_size,
        })
        self.rows = self.bytes = 0


@torch.inference_mode()
def predict_layers(model, input_ids, past_key_values=None, use_cache=False):
    """Run every layer; retain and project only the last position at each layer."""
    captured = [None] * len(model.model.layers)
    handles = []

    def hook(index):
        def save(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden[:, -1, :].detach().to("cpu", copy=True)
        return save

    try:
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(hook(index)))
        device = model.get_input_embeddings().weight.device
        output = model.model(
            input_ids=torch.tensor([input_ids], device=device),
            past_key_values=past_key_values, use_cache=use_cache,
            output_hidden_states=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    head = model.get_output_embeddings()
    norm = model.model.norm
    norm_device = next(norm.parameters()).device
    predictions = []
    for index, raw in enumerate(captured):
        normalized = (output.last_hidden_state[:, -1, :] if index == len(captured) - 1
                      else norm(raw.to(norm_device)))
        predictions.append(int(head(normalized.to(head.weight.device)).argmax(-1).item()))
    return predictions, output.past_key_values if use_cache else None


def collect_document(model, tokenizer, record, sequence_id, features, packed, args):
    """Generate with the final layer and label all layers at each next-token step."""
    prompt, truncated = document_prompt(record, tokenizer, args.context_size, args.max_new_tokens)
    generated = []
    past_key_values = None
    for generation_step in range(args.max_new_tokens):
        # Prefill the prompt once; cached decoding consumes only the latest token.
        step_input = prompt if generation_step == 0 else [generated[-1]]
        predictions, past_key_values = predict_layers(
            model, step_input, past_key_values=past_key_values, use_cache=True,
        )
        final_token = predictions[-1]
        features.write(pa.table({
            "sequence_id": pa.array([sequence_id] * len(predictions), type=pa.int64()),
            "generation_step": pa.array([generation_step] * len(predictions), type=pa.int32()),
            "position": pa.array([len(prompt) + generation_step - 1] * len(predictions), type=pa.int32()),
            "layer": pa.array(range(1, len(predictions) + 1), type=pa.int32()),
            "output_token_id": pa.array(predictions, type=pa.int32()),
            "label_match": pa.array([int(token == final_token) for token in predictions], type=pa.uint8()),
        }))
        generated.append(final_token)
        if final_token == tokenizer.eos_token_id:
            break
    packed.write(pa.table({
        "document_id": pa.array([str(record["id"])], type=pa.string()),
        "sequence_id": pa.array([sequence_id], type=pa.int64()),
        "input_ids": pa.array([prompt], type=pa.list_(pa.int32())),
        "generated_token_ids": pa.array([generated], type=pa.list_(pa.int32())),
        "truncated_article_tokens": pa.array([truncated], type=pa.int32()),
    }))
    return {"generated_tokens": len(generated), "truncated_documents": int(truncated > 0)}


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
        "context_size": args.context_size,
        "mode": "full_model_greedy_generation", "schema_version": 3,
        "num_layers": len(model.model.layers),
        "layer_numbering": "1-based decoder layers; final layer included",
        "prediction": "argmax(lm_head(final_norm(layer_output))) at last input position",
        "input": "join sequence_id; input_ids + generated_token_ids[:generation_step]",
        "decision_key": ["sequence_id", "generation_step"],
        "label_match": "1 if layer argmax equals final-layer argmax at this generation step; otherwise 0",
        "collection": "prompt prefill followed by KV-cached decoding; greedy, no logits processors",
        "use_cache": True,
        "prompt_template": "Article: {article}\\nSummary:",
        "truncation": "article tail truncated to reserve max_new_tokens within context_size",
        "documents": 0, "generated_tokens": 0,
        "truncated_documents": 0,
    }
    write_manifest(manifest_path, metadata)
    try:
        for sequence_id, record in enumerate(load_records(version, split, args)):
            counts = collect_document(model, tokenizer, record, sequence_id, features, packed, args)
            metadata["documents"] += 1
            for key, count in counts.items():
                metadata[key] += count
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
        choices=SPLITS, default=["train"],
        help="Splits to process (default: train); validation is an alias for val",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="cnn_dm")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens per document")
    parser.add_argument("--max-host-memory-gb", type=float, default=4)
    parser.add_argument("--memory-reserve-gb", type=float, default=1)
    parser.add_argument("--max-shard-mb", type=float, default=256)
    parser.add_argument("--max-documents", type=int, help="Optional limit per version/split")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--model-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = parser.parse_args()
    if not args.download_only and not args.model:
        parser.error("--model is required unless --download-only is supplied")
    for key in ("context_size", "max_new_tokens", "max_host_memory_gb", "max_shard_mb"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if not args.download_only and args.max_new_tokens >= args.context_size:
        parser.error("--max-new-tokens must be smaller than --context-size")
    if args.context_size < 2:
        parser.error("--context-size must be at least 2")
    if not math.isfinite(args.memory_reserve_gb) or args.memory_reserve_gb < 0:
        parser.error("--memory-reserve-gb must be finite and nonnegative")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be positive")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a filename prefix without directories")
    if len(set(args.versions)) != len(args.versions) or len(set(args.splits)) != len(args.splits):
        parser.error("--configs/--versions and --splits must not contain duplicates")
    args.reserve_bytes = int(args.memory_reserve_gb * 2**30)
    return args


def create_run_directory(output_dir):
    """Reserve a unique timestamped directory without overwriting previous runs."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f%z")
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in itertools.count():
        name = timestamp if suffix == 0 else f"{timestamp}_{suffix}"
        directory = output_dir / name
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory


def run():
    args = parse_args()
    args.output_dir = create_run_directory(args.output_dir)
    print(f"Run output directory: {args.output_dir.resolve()}", flush=True)
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


def main():
    started = time.perf_counter()
    completed = False
    try:
        run()
        completed = True
    finally:
        elapsed = time.perf_counter() - started
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        status = "finished" if completed else "stopped"
        print(
            f"Run {status}. Total elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d} "
            f"({elapsed:.2f} seconds)", flush=True,
        )


if __name__ == "__main__":
    main()
