"""Small offline checks: no model or dataset downloads required."""
from types import SimpleNamespace

import pyarrow as pa
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from create_layer_dataset import Sequence, document_sequences, extract_batch, export_parquet_csv


def test_run_directories_preserve_previous_outputs(tmp_path, monkeypatch):
    import create_layer_dataset as extractor
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


def test_document_sequences_preserves_short_tail():
    tokenizer = SimpleNamespace(
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        encode=lambda text, add_special_tokens: list(range(3, 13)),
    )
    sequences = list(document_sequences(
        {"id": "doc", "article": "article", "highlights": "summary"}, tokenizer, 11,
    ))
    assert [s.length for s in sequences] == [11, 1]
    assert [t for s in sequences for t in s.token_ids[:s.length]] == [1, *range(3, 13), 2]
    assert sequences[-1].token_ids == [2] + [0] * 10


def test_projection_default_groups_layers_per_token(monkeypatch):
    import sys
    from create_layer_dataset import parse_args

    monkeypatch.setattr(sys, "argv", ["create_layer_dataset.py", "--model", "example", "--output-dir", "example"])
    assert parse_args().projection_tokens == 1


@pytest.mark.parametrize("projection_tokens", [1, 2])
def test_predictions_match_independent_prefix_forwards(projection_tokens):
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0.0,
    )).eval()
    sequences = [
        Sequence("a", 0, 0, [1, 4, 5, 2], 4),
        Sequence("b", 1, 0, [1, 7, 0, 0], 2),
    ]

    class Collector:
        def __init__(self):
            self.tables = []

        def write(self, table):
            self.tables.append(table)

    features, packed = Collector(), Collector()
    extract_batch(model, sequences, features, packed,
                  SimpleNamespace(context_size=4, projection_tokens=projection_tokens))
    rows = pa.concat_tables(features.tables).to_pylist()
    assert len(rows) == 6 * 3
    assert set(rows[0]) == {"sequence_id", "position", "layer", "output_token_id"}
    assert packed.tables[0].to_pylist()[1]["input_ids"] == [1, 7]
    assert len({(r['sequence_id'], r['position'], r['layer']) for r in rows}) == len(rows)
    if projection_tokens == 1:
        assert [(r['sequence_id'], r['position'], r['layer']) for r in rows] == [
            (s.sequence_id, position, layer)
            for s in sequences for position in range(s.length) for layer in range(1, 4)
        ]

    with torch.inference_mode():
        for sequence in sequences:
            for position in range(sequence.length):
                inputs = torch.tensor([sequence.token_ids[:position + 1]])
                result = model.model(inputs, output_hidden_states=True, use_cache=False)
                for layer in range(1, 4):
                    hidden = result.hidden_states[layer][:, -1, :]
                    if layer < 3:
                        hidden = model.model.norm(hidden)
                    expected = model.lm_head(hidden).argmax(-1).item()
                    row = next(r for r in rows if (
                        r['sequence_id'], r['position'], r['layer']
                    ) == (sequence.sequence_id, position, layer))
                    assert row['output_token_id'] == expected
