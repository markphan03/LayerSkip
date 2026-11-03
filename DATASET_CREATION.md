# Creating CNN/DailyMail datasets

Use [`create_layer_dataset.py`](create_layer_dataset.py) to either save the original
dataset as Parquet or extract token-level features from every decoder layer.
Run the commands below from the repository root.

## Environment

Use a Python environment with the repository dependencies installed:

```bash
python -m pip install -r requirements.txt
```

The script imports `datasets`, `torch`, `transformers`, `numpy`, and `pyarrow`,
including in download-only mode. `pyarrow` is a dependency of `datasets`.
Model loading with automatic device placement also uses `accelerate`.

On the development machine, the existing environment is:

```bash
conda activate layer_skip
```

View all available options with:

```bash
python create_layer_dataset.py --help
```

## Dataset configurations and splits

The default dataset is `abisee/cnn_dailymail`. Each configuration contains
`train`, `validation`, and `test`; the configuration applies to the whole dataset.
See the [Hugging Face dataset metadata](https://huggingface.co/datasets/abisee/cnn_dailymail/blob/main/README.md).

| Selection | Default | Accepted values |
| --- | --- | --- |
| `--configs` | `3.0.0` | `1.0.0`, `2.0.0`, `3.0.0`, or aliases `v1`, `v2`, `v3` |
| `--splits` | All splits | `train`, `val` or `validation`, `test` |

`--versions` is an alias for `--configs`; `--split` is an alias for `--splits`.
You can supply multiple configurations and splits in one invocation. Output
directories use `v1`, `v2`, `v3`, and `val` regardless of the spelling you pass.

## Download a small test example

This saves three original records without loading a model or tokenizing text:

```bash
python create_layer_dataset.py \
  --download-only \
  --split test \
  --max-documents 3 \
  --output-dir ./cnn_test_example
```

Expected files:

```text
cnn_test_example/
└── v3/
    └── test/
        ├── cnn_dm_test_001.parquet
        └── manifest.json
```

Each Parquet row has three string columns: `id`, `article`, and `highlights`.
The manifest records the configuration, split, document count, completion status,
and shard sizes. The three-record example has been run successfully; the saved
Parquet was checked for three rows and these columns. Layer extraction has not
yet been validated with a model.

Read the example:

```python
import json
from pathlib import Path

import pyarrow.parquet as pq

root = Path("cnn_test_example/v3/test")
manifest = json.loads((root / "manifest.json").read_text())
assert manifest["status"] == "complete"
table = pq.read_table(root / "cnn_dm_test_001.parquet")
print("Records:", table.num_rows)
print(table.to_pylist()[0])
```

## Download complete splits

Remove `--max-documents` to save the entire test split:

```bash
python create_layer_dataset.py \
  --download-only \
  --configs 3.0.0 \
  --split test \
  --output-dir ./cnn_test_full
```

To save every configuration and split:

```bash
python create_layer_dataset.py \
  --download-only \
  --configs 1.0.0 2.0.0 3.0.0 \
  --splits train validation test \
  --output-dir ./cnn_all
```

The script streams records from Hugging Face. `--max-documents` limits records
processed per configuration and split; it does not impose a network byte limit.
Use a fresh output directory for each run. Existing configuration/split output
directories cause an error, including those left by an interrupted run. There is
currently no resume or overwrite option.

## Extract layer features

Omit `--download-only` and supply `--model`. For a small initial extraction:

```bash
python create_layer_dataset.py \
  --model facebook/layerskip-llama2-7B \
  --configs 3.0.0 \
  --split test \
  --max-documents 3 \
  --context-size 256 \
  --batch-size 1 \
  --projection-tokens 8 \
  --output-dir ./cnn_layer_example
```

The example model requires Hugging Face access. `--model` also accepts a local
model directory. The script expects a Llama-compatible model exposing decoder
layers, final normalization, and an output embedding head. Automatic device
placement is the default; specify `--device cpu` or `--device cuda:0` to select
a device. Disk/meta model offloading is unsupported.

Extraction streams its input from Hugging Face independently of any earlier
download-only export; there is currently no local Parquet input option.
`--max-documents` limits articles, so three documents can produce many sequences
and feature rows. This mode uses teacher forcing, not generated summaries.

### Tokenization and batching

Each document is formatted as:

```text
[Prompt]
Summarize the following document:
<article>

[Answer]
Answer:
<summary><EOS>
```

A BOS token is prepended when the tokenizer defines one. Tokens are chunked into
sequences of `--context-size`:

- Full chunks are kept.
- Trailing chunks below 10% of the context size are dropped.
- Chunks from 10% to below 90% repeat tokens from the beginning of the summary
  until reaching at least 90%, then pad to the context size.
- Chunks already at or above 90% are padded to the context size.

Synthetic summary repetitions are marked. Padding has attention mask zero and
produces no feature rows. Each sequence belongs to one document; batches combine
sequences from different documents when needed. The final batch can be smaller
than the requested batch size.

### Output and labels

Under `cnn_layer_example/v3/test/`, extraction writes:

| File | Contents |
| --- | --- |
| `cnn_dm_test_001.parquet`, subsequent shards | One row per non-padding token position and decoder layer |
| `cnn_dm_sequences_test_001.parquet`, subsequent shards | Token IDs, next-token targets, masks, document IDs, and batch placement for each sequence |
| `manifest.json` | Completion status, arguments, dimensions, dtypes, label definitions, and shard lists |

Join feature rows to sequence rows using `sequence_id` within the same
configuration and split. Decoder layer numbers start at 1; embeddings are excluded.

| Feature | Meaning |
| --- | --- |
| `token_id` | Input token at this position |
| `real_target` | Next input token; `-100` when no target is defined |
| `hidden_output` | Raw decoder output before final normalization |
| `distribution` | Probabilities after applying final normalization and the LM head |
| `kl_to_final` | `KL(final_distribution || layer_distribution)` using natural logs |
| `top10_token_ids`, `top10_probabilities` | Highest-probability tokens at this layer |
| `layer_prediction` | This layer's argmax token |
| `full_model_prediction` | Final layer's argmax token |
| `label_match` | 1 when this layer's argmax matches the final argmax |
| `label_first_match` | 1 only at the first matching layer for this position |
| `label_alpha` | 1 when the final argmax token has sufficiently high probability at this layer |

The alpha label uses:

```text
p_layer(final_argmax) >= alpha * max(p_layer)
```

By default, alpha depends on the final model's argmax probability `p`:

```text
p <= 0.2:       alpha = 0.6
0.2 < p < 0.9: alpha = 0.6 + (p - 0.2) * (0.3 / 0.7)
p >= 0.9:      alpha = p
```

For a fixed threshold, pass `--alpha-mode fixed --alpha 0.8`.
Labels are stored as unsigned 8-bit integers; KL values and scalar probabilities
use float32. Token IDs use int32. See the Parquet schema for all columns.

### Decode vector columns

Hidden outputs and full distributions are binary arrays, with dimensions and
dtypes recorded in the manifest. Defaults are little-endian float16 hidden
outputs and float32 distributions. Read one row without loading an entire shard:

```python
import numpy as np
import pyarrow.parquet as pq

parquet = pq.ParquetFile("cnn_layer_example/v3/test/cnn_dm_test_001.parquet")
row = next(parquet.iter_batches(batch_size=1)).to_pylist()[0]
hidden = np.frombuffer(row["hidden_output"], dtype="<f2")
probabilities = np.frombuffer(row["distribution"], dtype="<f4")
print(hidden.shape, probabilities.shape)
```

Adjust the decoding dtype if you select `--hidden-dtype` or
`--distribution-dtype`. For `--hidden-dtype float8_e4m3fn`, decode with:

```python
import torch

hidden = torch.frombuffer(
    bytearray(row["hidden_output"]), dtype=torch.uint8,
).view(torch.float8_e4m3fn).float()
```

## Memory and storage controls

| Option | Default | Purpose |
| --- | --- | --- |
| `--context-size` | `1024` | Tokens per sequence; must fit the model's context limit |
| `--batch-size` | `2` | Maximum requested sequences per batch |
| `--projection-tokens` | `16` | Token positions projected through the LM head at once |
| `--max-host-memory-gb` | `4` | Budget used for extraction working-memory estimates |
| `--memory-reserve-gb` | `1` | Host memory reserve used by capacity checks |
| `--max-shard-mb` | `256` | Shard rotation target based on uncompressed Arrow bytes |
| `--hidden-dtype` | `float16` | `float8_e4m3fn`, `float16`, or `float32` storage |
| `--distribution-dtype` | `float32` | `float16` or `float32` storage |

Memory and shard units use powers of 1024. The host memory budget is an estimate,
not a cap on total process memory or model weights. Raw layer outputs are
offloaded to CPU after each layer; batch capacity is adjusted to available host
memory. GPU memory is not automatically budgeted. If extraction runs out of
memory, reduce batch size, context size, or projection tokens and retry with a
fresh output directory.

Full vocabulary distributions are saved at every layer and token position, so
feature exports can be large even for a small number of articles. Float16
distribution storage reduces size but can round very small probabilities to
zero; labels and KL values are computed before storage conversion. Shards are
Zstandard-compressed, and actual file sizes can differ from the rotation target.

Use `--model-revision` and `--dataset-revision` with specific commit IDs when you
need to pin the inputs; both default to `main`.
