# CNN/DailyMail dataset generator

[`generate_cnn_dataset.py`](generate_cnn_dataset.py) builds a Parquet dataset of
raw decoder hidden states, next-token predictions, and full vocabulary
probabilities. By default, it extracts teacher-forced predictions from
[`facebook/layerskip-llama2-7B`](https://huggingface.co/facebook/layerskip-llama2-7B)
for exactly the first three documents of
[`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail),
configuration `3.0.0`, test split. It uses `article` as the prompt and `highlights`
as the reference summary. The reference summary is included in the model input;
the script records predictions at each position rather than generating a new
summary autoregressively.

## Quick start

Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python generate_cnn_dataset.py --output cnn_layer_outputs.parquet
```

The runner selects CUDA when available, otherwise CPU. Override with
`--device cuda:0` or `--device cpu`. The complete model runs on one device, using
float16 on CUDA and float32 on CPU. `--batch-size` defaults to 2 and
`--max-length` to 4096, capped at the model's context length. Hugging Face model
access and sufficient memory for the 7B weights are required for this command.
Importing the module does not download data or weights.

The command saves `cnn_layer_outputs.parquet`, reads it back, and prints the
column names, tensor shapes for the first document, and verification results.

## Command-line options

All defaults are defined directly in the script's argument parser. View its
options with:

```bash
python generate_cnn_dataset.py --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `abisee/cnn_dailymail` | Dataset ID. Records must provide `article` and `highlights`. |
| `--max-documents` | `3` | Total documents to extract. The current parser accepts only `3`; the writer and verifier enforce this count. |
| `--start-row` | `0` | Zero-based starting row. Skip this many records before selecting three; must be non-negative. |
| `--version` | `v3` | Dataset configuration: `v1` = `1.0.0`, `v2` = `2.0.0`, `v3` = `3.0.0`. |
| `--split` | `test` | Dataset split: `train`, `val` (validation), or `test`. |
| `--batch-size` | `2` | Maximum documents per forward pass; must be positive. |
| `--max-length` | `4096` | Maximum input tokens, including BOS, markers, summary, and EOS; must be positive. Capped at the model's context length. |
| `--model-path` | `facebook/layerskip-llama2-7B` | Llama-compatible model ID or local checkpoint directory. |
| `--output` | `cnn_layer_outputs.parquet` | Output path. Parent directories are created automatically. |
| `--device` | `cuda` if available, otherwise `cpu` | Device for the complete model, for example `cuda:0` or `cpu`. |

For an explicit configuration using the requested model and dataset:

```bash
python generate_cnn_dataset.py \
    --dataset abisee/cnn_dailymail \
    --version v3 \
    --split test \
    --start-row 0 \
    --max-documents 3 \
    --batch-size 2 \
    --max-length 4096 \
    --model-path facebook/layerskip-llama2-7B \
    --output cnn_layer_outputs.parquet
```

Changing `--start-row` selects a different consecutive set of three documents.
For example, `--start-row 10` selects rows 10, 11, and 12, using zero-based row
numbers. The stream keeps dataset order and does not shuffle. If fewer than
three records remain, the writer raises an error.

The `--model-path` option uses the same extraction code: the model must expose
the Llama-style embedding, decoder layers, final normalization, and LM head.
The loader also expects the selected dataset to support the chosen configuration
and split.

## Batch size and document count

`max_documents` controls the total number of documents loaded. `batch_size`
controls how many of those documents can be processed together. Batch size can
be larger than the document count:

| `max_documents` | `batch_size` | Actual batches | Parquet rows |
| --- | --- | --- | --- |
| 3 | 1 | 1, 1, 1 | 3 |
| 3 | 2 | 2, 1 | 3 |
| 3 | 3 | 3 | 3 |
| 3 | 16 | 3 | 3 |

For example:

```bash
python generate_cnn_dataset.py --batch-size 16 --max-documents 3
```

This processes one batch of three documents. It does not duplicate documents or
add thirteen empty examples. `drop_last=False` keeps incomplete batches. Within
each batch, token sequences are left padded to the longest sequence in that
batch; those padding tokens are removed before serialization.

## Input and prediction positions

Each input uses the tokenizer's actual BOS and EOS IDs:

```text
BOS [Prompt] article [Summary] reference highlights EOS
```

Markers and text segments are tokenized separately without automatic special
tokens. Only the article is truncated, including to zero tokens if necessary.
An error is raised if BOS, both markers, the entire summary, and EOS cannot fit.
Batches are left padded, using EOS as padding when the tokenizer has no pad ID.

Labels use the standard **unshifted** causal-LM convention: they contain the
summary and EOS IDs at those tokens' own input positions; all other positions,
including padding, contain `-100`.

For each labeled input position `k`, extraction selects the raw hidden state at
`k - 1`. These states start at the last summary-marker token and end at the last
summary token. They predict `t1, ..., tM, EOS`, giving `T = M + 1`. The EOS hidden
state is excluded. An empty summary has one prediction position, for EOS.

| Source hidden state | Predicted token / reference target |
| --- | --- |
| Last token immediately before `t1` | `t1` |
| `t1` | `t2` |
| ... | ... |
| `tM` | EOS |

The existing sequential decoder loop runs in evaluation mode with gradients
disabled. It stores each decoder layer's raw output and computes
`softmax(lm_head(final_norm(raw)))`, with exactly one final normalization per
selected state. Normalized states are not passed into the next decoder layer.
Predictions are argmax IDs of the stored probabilities. After all layers run,
matching flags compare each prediction with the reference ID **or** the final
layer's prediction at the same position; final-layer flags are always true:

```python
# prediction_ids: [L, T]; reference_target_ids: [T]
matches_reference_or_final = (
    prediction_ids == reference_target_ids[None, :]
) | (
    prediction_ids == prediction_ids[-1:]
)
```

`prediction_ids` always means model predictions. `reference_target_ids` always
means the actual summary token IDs followed by EOS.

## Parquet layout and reconstruction

There are exactly three rows and eight columns, one document per row group.
Padding is removed before storage. All axes are nested Arrow large lists (64-bit
offsets to support large distributions), ordered layer, prediction position,
then feature/vocabulary dimension where applicable.
Layer index 0 is the first decoder layer; the embedding layer is excluded.

| Column | Tensor shape | Storage / reconstructed PyTorch dtype |
| --- | --- | --- |
| `input_ids` | `[S]` | int64 / `torch.int64` |
| `labels` | `[S]` | int64 / `torch.int64` |
| `attention_mask` | `[S]` | int64 / `torch.int64` (all ones) |
| `reference_target_ids` | `[T]` | int64 / `torch.int64` |
| `hidden_outputs` | `[L, T, H]` | float32 / `torch.float32` |
| `prediction_ids` | `[L, T]` | int64 / `torch.int64` |
| `matches_reference_or_final` | `[L, T]` | Boolean / `torch.bool` |
| `distributions` | `[L, T, V]` | float32 / `torch.float32` |

`S` is the unpadded input length, `T = M + 1` counts summary tokens plus EOS,
`L` is the decoder-layer count, `H` the hidden dimension, and `V` the complete
vocabulary size. Hidden states computed in float16 are converted to float32
for storage; this preserves their computed
values but does not add precision. Softmax is computed in float32. All vocabulary
probabilities are retained without top-k filtering or quantization. The two
float columns occupy `4 * L * T * (H + V)` bytes per document before compression.
Zstandard compression is lossless. Arrow schema metadata records axes, dtypes,
and prediction alignment.

Extraction retains the output tensors for one batch in host memory at a time.
Smaller batches reduce that memory use when they contain fewer actual documents.
Output is first written to a `.parquet.partial` file. After the writer closes
and confirms the document count, it replaces the requested output file. A
successful run therefore overwrites an existing file at that output path.

To reconstruct the tensors:

```python
import pyarrow.parquet as pq
from generate_cnn_dataset import reconstruct_row

# Read one document at a time to avoid loading all full-vocabulary arrays at once.
with pq.ParquetFile("cnn_layer_outputs.parquet") as parquet:
    for batch in parquet.iter_batches(batch_size=1):
        tensors = reconstruct_row(batch)
        print({name: tuple(value.shape) for name, value in tensors.items()})
        # tensors["distributions"][layer, position, token_id] is a probability.
```

`reconstruct_row(table, row_index)` also accepts an already loaded PyArrow Table.
It creates writable CPU tensors directly from Arrow arrays, avoiding large
intermediate Python float lists.

## Automatic verification

The runner calls `verify_parquet(output_path, tokenizer.eos_token_id)` after
saving. It reads one document at a time and checks:

- The file contains exactly three rows and the eight expected columns.
- Input IDs, labels, and attention masks have the same unpadded length.
- Labels contain the actual summary and EOS IDs at their unshifted positions.
- Hidden states, predictions, reference targets, matching flags, and
  distributions all use the same `T = M + 1` prediction positions.
- Matching flags satisfy the reference-or-final comparison, including all-true
  final-layer flags.
- Probabilities are finite, non-negative, sum to one within numerical tolerance,
  and have argmax IDs equal to `prediction_ids`.

It also prints the column names and reconstructed tensor shapes for the first
document. The independent model comparisons described below check the raw
hidden-state values and final normalization during testing.

## Offline tests

```bash
python -m pytest tests/test_generate_cnn_dataset.py -q
```

[`tests/test_generate_cnn_dataset.py`](tests/test_generate_cnn_dataset.py) uses
a tiny randomly initialized Hugging Face Llama and a deterministic tokenizer.
The tests cover both eager and SDPA attention, unshifted labels, article-only
truncation, empty summaries, raw outputs from every decoder layer, parity with
the normal Hugging Face logits and loss, left-padding invariance, both matching
conditions, exact document count, malformed alignment, and lossless Parquet
reconstruction. Model and dataset downloads are replaced only in the runner
test; decoder computation and PyArrow serialization run normally.
