"""Small offline checks: no model or dataset downloads required."""
from types import SimpleNamespace

import pyarrow as pa
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from create_layer_dataset_v2 import document_prompt, predict_layers, collect_document, export_parquet_csv


def test_run_directories_preserve_previous_outputs(tmp_path, monkeypatch):
    import create_layer_dataset_v2 as extractor
    from datetime import datetime, timezone

    class FixedDatetime:
        @staticmethod
        def now():
            return datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(extractor, "datetime", FixedDatetime)
    root = tmp_path / "outputs"
    previous = root / "v3" / "train"
    previous.mkdir(parents=True)
    marker = previous / "existing.csv"
    marker.write_text("previous data")
    first = extractor.create_run_directory(root)
    second = extractor.create_run_directory(root)
    assert first.parent == second.parent == root
    assert first.is_dir() and second.is_dir()
    assert first != second
    assert second.name == first.name + "_1"
    assert marker.read_text() == "previous data"


def test_csv_preserves_rows_and_token_lists(tmp_path):
    import csv
    import json
    import pyarrow.parquet as pq

    path = tmp_path / "sequences.parquet"
    pq.write_table(pa.table({
        "document_id": ['doc, "one"', "second"],
        "input_ids": [[1, 4, 2], [1, 2]],
    }), path)
    destination = export_parquet_csv(path)
    with destination.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r['document_id'] for r in rows] == ['doc, "one"', 'second']
    assert [json.loads(r['input_ids']) for r in rows] == [[1, 4, 2], [1, 2]]


class Collector:
    def __init__(self):
        self.tables = []

    def write(self, table):
        self.tables.append(table)

    def rows(self):
        return pa.concat_tables(self.tables).to_pylist()


def tokenizer_stub():
    def encode(text, add_special_tokens=False):
        return {"Article: ": [3], "\nSummary:": [4], "article": [5, 6, 7, 8]}[text]
    return SimpleNamespace(bos_token_id=1, eos_token_id=2, encode=encode)


def test_prompt_excludes_reference_and_preserves_answer_cue():
    record = {"id": "doc", "article": "article", "highlights": "never tokenize this"}
    prompt, truncated = document_prompt(record, tokenizer_stub(), 8, 3)
    assert prompt == [1, 3, 5, 6, 4]
    assert truncated == 2
    with pytest.raises(ValueError):
        document_prompt(record, tokenizer_stub(), 5, 3)


def test_predict_layers_matches_hidden_states_and_removes_hooks():
    torch.manual_seed(0)
    model = tiny_model()
    inputs = [1, 4, 5]
    predictions, hidden_vectors, _ = predict_layers(model, inputs)
    raw_final = []
    handle = model.model.norm.register_forward_pre_hook(
        lambda module, inputs: raw_final.append(inputs[0].detach().clone())
    )
    with torch.inference_mode():
        result = model.model(torch.tensor([inputs]), output_hidden_states=True, use_cache=False)
        expected = []
        for index in range(1, 4):
            hidden = result.hidden_states[index][:, -1, :]
            if index < 3:
                hidden = model.model.norm(hidden)
            expected.append(model.lm_head(hidden).argmax(-1).item())
    handle.remove()
    assert predictions == expected
    expected_raw = [state[0, -1].float() for state in result.hidden_states[1:-1]]
    expected_raw.append(raw_final[0][0, -1].float())
    torch.testing.assert_close(torch.tensor(hidden_vectors), torch.stack(expected_raw))
    assert all(not layer._forward_hooks for layer in model.model.layers)


def tiny_model():
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
    )).eval()


@pytest.mark.parametrize("last_token", [2, 9])
def test_generation_uses_final_token_and_labels_every_match(monkeypatch, last_token):
    import create_layer_dataset_v2 as extractor
    contexts = []
    outputs = iter([[10, 11, 10], [12, 13, 13], [last_token] * 3])

    def predict(model, context, past_key_values=None, use_cache=False):
        assert use_cache
        if past_key_values is not None:
            assert len(context) == 1
        full_context = (past_key_values or []) + list(context)
        contexts.append(full_context)
        return next(outputs), [[0.1, 0.2]] * 3, full_context

    monkeypatch.setattr(extractor, "predict_layers", predict)
    features, packed = Collector(), Collector()
    args = SimpleNamespace(context_size=32, max_new_tokens=3)
    counts = collect_document(None, tokenizer_stub(), {"id": "doc", "article": "article"},
                              0, features, packed, args)
    prompt = [1, 3, 5, 6, 7, 8, 4]
    assert contexts == [prompt, prompt + [10], prompt + [10, 13]]
    rows = features.rows()
    assert [r['label_match'] for r in rows] == [1, 0, 1, 0, 1, 1, 1, 1, 1]
    assert packed.rows()[0]['generated_token_ids'] == [10, 13, last_token]
    assert len(rows) == 9
    assert counts['generated_tokens'] == 3
    assert all(rows[index]['label_match'] == 1 for index in (2, 5, 8))
    for row in rows:
        stored = packed.rows()[0]
        context = stored['input_ids'] + stored['generated_token_ids'][:row['generation_step']]
        assert context == contexts[row['generation_step']]
        assert row['position'] == len(context) - 1


def test_eos_stops_generation_after_recording_all_layers(monkeypatch):
    import create_layer_dataset_v2 as extractor
    contexts = []

    def predict(model, context, **kwargs):
        contexts.append(context)
        return [7, 2, 2], [[0.1, 0.2]] * 3, None

    monkeypatch.setattr(extractor, "predict_layers", predict)
    features, packed = Collector(), Collector()
    collect_document(None, tokenizer_stub(), {"id": "doc", "article": "article"}, 0,
                     features, packed, SimpleNamespace(context_size=32, max_new_tokens=5))
    assert len(contexts) == 1
    assert len(features.rows()) == 3
    assert packed.rows()[0]['generated_token_ids'] == [2]


def test_greedy_output_matches_full_model_generation():
    torch.manual_seed(4)
    model = tiny_model()
    tokenizer = tokenizer_stub()
    args = SimpleNamespace(context_size=32, max_new_tokens=5)
    record = {"id": "doc", "article": "article"}
    features, packed = Collector(), Collector()
    collect_document(model, tokenizer, record, 0, features, packed, args)
    prompt, _ = document_prompt(record, tokenizer, 32, 5)
    generated = []
    with torch.inference_mode():
        for _ in range(5):
            logits = model(torch.tensor([prompt + generated]), use_cache=False).logits
            token = logits[0, -1].argmax().item()
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
    assert packed.rows()[0]['generated_token_ids'] == generated
    rows = features.rows()
    for index in range(0, len(rows), 3):
        assert [r['layer'] for r in rows[index:index + 3]] == [1, 2, 3]


def test_split_writes_manifest_parquet_and_csv(tmp_path, monkeypatch):
    import csv
    import json
    import sys
    import pyarrow.parquet as pq
    import create_layer_dataset_v2 as extractor

    monkeypatch.setattr(sys, 'argv', [
        'create_layer_dataset_v2.py', '--model', 'offline-test',
        '--output-dir', str(tmp_path), '--context-size', '32',
        '--max-new-tokens', '2',
    ])
    args = extractor.parse_args()
    monkeypatch.setattr(extractor, 'load_records', lambda *args: iter([
        {'id': 'doc', 'article': 'article', 'highlights': 'excluded'},
    ]))
    monkeypatch.setattr(extractor, 'predict_layers', lambda *args, **kwargs: ([2, 3, 2], [[0.1] * 16] * 3, None))
    extractor.process_split(tiny_model(), tokenizer_stub(), 'v3', 'train', args)
    root = tmp_path / 'v3' / 'train'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['schema_version'] == 4
    assert manifest['documents'] == manifest['generated_tokens'] == 1
    for key in ('feature_shards', 'sequence_shards'):
        for shard in manifest[key]:
            rows = pq.read_table(root / shard['file']).to_pylist()
            with (root / shard['csv_file']).open(encoding='utf-8-sig', newline='') as stream:
                csv_rows = list(csv.DictReader(stream))
            assert len(rows) == len(csv_rows) == shard['rows']
            if key == 'feature_shards':
                assert manifest['hidden_size'] == 16
                assert len(rows[0]['hidden_output']) == 16
                assert json.loads(csv_rows[0]['hidden_output']) == rows[0]['hidden_output']
                assert pq.read_table(root / shard['file']).schema.field('hidden_output').type == pa.list_(pa.float32(), 16)
                assert [r['label_match'] for r in rows] == [1, 0, 1]
                assert [r['label_match'] for r in csv_rows] == ['1', '0', '1']


def test_cached_predictions_and_labels_match_full_prefix_at_every_layer():
    torch.manual_seed(12)
    model = tiny_model()
    prefix = [1, 5, 7, 9]
    cache = None
    for step in range(5):
        step_input = prefix if step == 0 else [prefix[-1]]
        cached, cached_hidden, cache = predict_layers(model, step_input, past_key_values=cache, use_cache=True)
        full, full_hidden, _ = predict_layers(model, prefix)
        assert cached == full
        torch.testing.assert_close(torch.tensor(cached_hidden), torch.tensor(full_hidden), atol=1e-6, rtol=1e-5)
        assert [int(t == cached[-1]) for t in cached] == [int(t == full[-1]) for t in full]
        assert cache.get_seq_length() == len(prefix)
        prefix = prefix + [cached[-1]]
    # A new document must begin with a new cache, independent of the previous one.
    fresh, _, fresh_cache = predict_layers(model, [1, 3], use_cache=True)
    reference, _, _ = predict_layers(model, [1, 3])
    assert fresh == reference
    assert fresh_cache.get_seq_length() == 2


@pytest.mark.parametrize('fails', [False, True])
def test_main_prints_elapsed_time_even_on_failure(monkeypatch, capsys, fails):
    import create_layer_dataset_v2 as extractor
    ticks = iter([10.0, 75.25])
    monkeypatch.setattr(extractor.time, 'perf_counter', lambda: next(ticks))

    def run():
        if fails:
            raise RuntimeError('test failure')

    monkeypatch.setattr(extractor, 'run', run)
    if fails:
        with pytest.raises(RuntimeError, match='test failure'):
            extractor.main()
    else:
        extractor.main()
    output = capsys.readouterr().out
    assert '00:01:05 (65.25 seconds)' in output
    assert ('Run stopped.' if fails else 'Run finished.') in output
