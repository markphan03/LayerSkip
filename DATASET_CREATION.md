# Creating CNN/DailyMail layer predictions

Run with the repository dependencies installed (`pip install -r requirements.txt`).
The default is CNN/DailyMail configuration 3.0.0, training split.

```bash
python create_layer_dataset.py \
  --model facebook/layerskip-llama2-7B \
  --split train \
  --max-documents 3 \
  --batch-size 1 \
  --output-dir cnn_layer_example
```

Remove `--max-documents` to process the entire split. Model access may require
Hugging Face authentication. `--model` also accepts a local model directory.
Each invocation creates a timestamped run folder under `--output-dir`, using
local time with microseconds and a UTC offset. The script prints its location.
You can reuse the same command without deleting earlier results. All selected
configurations and splits for one invocation share that run folder. This also
applies to download-only runs. Existing outputs are left untouched.

The script feeds `Article: <article>\nSummary: <reference highlights>` plus BOS
(if defined) and EOS through every decoder layer. This is teacher-forced
extraction, not generated summaries. It splits tokens into independent contexts
of at most `--context-size` (default 1024). Earlier chunks are not visible to
later chunks. Every tail is retained; no synthetic tokens are added. Batch
padding is masked and produces no prediction rows.

Under `cnn_layer_example/<timestamp>/v3/train/`:

| File | Columns |
| --- | --- |
| `cnn_dm_train_*.parquet` | `sequence_id`, `position`, `layer`, `output_token_id` |
| `cnn_dm_sequences_train_*.parquet` | `document_id`, `sequence_id`, `chunk_index`, `input_ids` |
| `manifest.json` | Model/dataset settings, completion status, counts, shard lists |

Every Parquet shard also has a matching `.csv` copy for opening in Excel,
Numbers, or another spreadsheet application. CSV files contain the same columns
and rows; input-token lists are written as JSON in a single cell. Token values
are numeric IDs, not decoded words. Large exports may exceed a spreadsheet's row
limit; Parquet remains the primary format for processing the full dataset.

Each prediction row records the greedy next-token prediction from one decoder
layer. The script applies the model's final normalization and shared output head
to that layer's hidden state. Layer numbers start at 1 and include the final
layer. No labels, hidden vectors, or probability vectors are saved.

Join on `sequence_id` within a split. The exact token context for a prediction is:

```python
input_prefix = sequence_row['input_ids'][:prediction_row['position'] + 1]
output_token_id = prediction_row['output_token_id']
```

`position` is zero-based. The input token at this position is already consumed;
`output_token_id` predicts the following token. The final position, including
EOS, also receives a prediction. The full sequence is stored once for compactness;
future tokens after `position` are not part of that prediction's input.
For later labeling, compare outputs at the same `(sequence_id, position)` with
the row whose `layer` equals the manifest's `num_layers`.

Omit `--projection-tokens` to use the default of 1: predictions are written in
sequence order, then position order, then layer order. For example, position 0
has layers 1 through 32 together, followed by position 1 with layers 1 through
32. This applies to both Parquet and CSV. You do not need to pass the option.
Explicitly setting `--projection-tokens` above 1 projects multiple positions
together and writes those positions for each layer before moving to the next
group. It changes processing and row order, not which predictions are collected.
All raw layer outputs for one batch are temporarily kept
in host RAM. `--max-host-memory-gb` and `--memory-reserve-gb` guide batch-size
estimates; they do not cap total process or GPU memory. Reduce batch/context size
if needed. `--max-shard-mb` controls rotation based on uncompressed Arrow size.

Other options include `--configs` (`1.0.0`, `2.0.0`, `3.0.0`), `--splits`
(`train`, `validation`, `test`), device and model dtype. Model/dataset revisions
default to `main`; supply commit IDs for reproducibility. Use `--help` for details.

`--download-only` saves original `id`, `article`, and `highlights` rows without
loading a model. Extraction streams from Hugging Face directly, independently
of any download-only export.
