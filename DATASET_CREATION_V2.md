# Collecting labeled layer predictions during generation

`create_layer_dataset_v2.py` runs every layer for each next-token decision,
records each layer's prediction, and appends the final layer's chosen token.
The original `create_layer_dataset.py` retains teacher-forced collection.

```bash
./.venv/bin/python create_layer_dataset_v2.py \
  --model facebook/layerskip-llama2-7B \
  --split train \
  --max-documents 1 \
  --context-size 1024 \
  --max-new-tokens 16 \
  --device cpu \
  --output-dir layer_example
```

Each run creates a new timestamped folder. Existing runs are untouched.
The model/tokenizer must be accessible through Hugging Face or a local model path.
CPU collection on a 7B model can be slow.

The prompt is `Article: <article>\nSummary:` with BOS if defined. Reference
highlights are never supplied. `--context-size` limits prompt plus generated
output. Article tokens beyond the available budget are dropped from the end,
preserving the answer cue and reserving `--max-new-tokens` slots. Truncation is
printed and recorded. Use a larger context within the model limit to preserve
more of the article.

For every generated token:

1. Run the prompt through all decoder layers once and retain its KV cache.
   On subsequent steps, process only the newly appended token using that cache.
2. Apply final normalization and the shared vocabulary head to the last position
   at each layer, recording its highest-scoring token ID.
3. Set `label_match` to 1 for every layer whose token equals the final layer's
   token, otherwise 0. All matches receive 1, including the final layer itself.
4. Append the final layer's token and repeat until EOS or the generation limit.

There is no early exit, drafting, speculative verification, sampling, or logits
processing. A separate KV cache is used for each document and discarded afterward.
Caching reuses earlier attention keys and values, trading additional memory for
less repeated computation. All layers still run for each new token, and labels
are calculated the same way. It collects full-model-generated
prefixes, not rejected draft paths. Only the last hidden position is retained
per layer; input-article positions are not exported as separate predictions.

Under `<output-dir>/<timestamp>/v3/train/`, both Parquet and CSV are saved:

| Files | Contents |
| --- | --- |
| `cnn_dm_train_*` | One row per generation step and layer |
| `cnn_dm_sequences_train_*` | One row per document with its prompt and generated output |
| `manifest.json` | Schema version 4, settings, hidden dimension/dtype, counts, truncation and shard lists |

Prediction columns:

- `sequence_id`: document index within this configuration/split.
- `generation_step`: zero-based next-token generation step.
- `position`: last position of that step's input, zero-based.
- `layer`, `output_token_id`: 1-based layer number and its prediction.
- `hidden_output`: raw decoder output vector at the last input position, before
  final normalization and the vocabulary head. The final layer also stores its
  raw output. Parquet uses a fixed-size float32 list of `hidden_size` elements;
  CSV uses a JSON list in one cell. Values computed at lower model precision
  are converted to float32 for storage, without recovering lost precision.
- `label_match`: 1 if the prediction matches the final layer at the same step, else 0.

Rows are grouped by document, generation step, then layer. For example, step 0
contains layers 1 through 32 together, followed by step 1's layers 1 through 32.
The label measures agreement with the final model, not correctness against a
reference summary. It is offline supervision, not an inference-time feature.

The sequences file stores `document_id`, `sequence_id`, `input_ids` (the prompt),
`generated_token_ids` (including EOS if generated), and `truncated_article_tokens`.
Lists are JSON cells in CSV. Numeric IDs are not decoded into text.
Hidden vectors substantially increase output size. For hidden size 4096,
32 layers and 16 generated tokens, vectors alone occupy 8 MiB before compression.

Join on `sequence_id` to reconstruct a prediction's exact input:

```python
context = sequence_row['input_ids'] + sequence_row['generated_token_ids'][:prediction_row['generation_step']]
```

Group by `(sequence_id, generation_step)` when comparing layer predictions.

At program exit, the terminal prints total elapsed wall-clock time as
`HH:MM:SS` and seconds. This includes model loading/downloads, dataset loading,
generation, and writing Parquet/CSV files. An interrupted or failed run also
prints elapsed time with `Run stopped` when Python can execute cleanup.

V2 processes one document and one generation step at a time. It has no
`--exit-layer`, `--num-speculations`, `--batch-size`, or `--projection-tokens`
option. Memory budgets are not enforced for model inference;
`--max-host-memory-gb` only bounds the download-only buffer.
`--memory-reserve-gb` and `--max-shard-mb` guide output shard rotation.
`--download-only`, configuration/split selection, revisions, device and dtype
options remain available. Use `--help` for details.
