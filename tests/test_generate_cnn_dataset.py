"""Offline extraction tests: a tiny random Llama, no model/dataset downloads."""

from types import SimpleNamespace

from datasets import IterableDataset
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import generate_cnn_dataset as extraction


class WordTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None  # Exercise EOS-as-padding without masking the real EOS.
    vocabulary = {
        "[Prompt]": 3, "[Summary]": 4, "alpha": 5, "beta": 6,
        "gamma": 7, "delta": 8, "epsilon": 9, "one": 10, "two": 11, "three": 12,
    }

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [self.vocabulary[word] for word in text.split()]

    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["padding"] is False and kwargs["truncation"] is False
        return {"input_ids": [self.encode(text) for text in texts]}


@pytest.fixture
def tokenizer():
    return WordTokenizer()


@pytest.fixture
def records():
    return [
        {"article": "alpha beta gamma delta epsilon", "highlights": "one two three"},
        {"article": "beta", "highlights": "two"},
        {"article": "gamma delta", "highlights": ""},
    ]


@pytest.fixture(params=["eager", "sdpa"])
def model(request):
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.2,
        bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = request.param
    model = LlamaForCausalLM(config)
    # Unequal norm weights make an accidental second normalization observable.
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.5, 1.5, config.hidden_size))
    return model


def test_unshifted_labels_and_left_padding(tokenizer, records):
    batch = extraction.TokenEncoder(tokenizer, max_length=32)(records)
    for index, record in enumerate(records):
        summary = tokenizer.encode(record["highlights"]) + [tokenizer.eos_token_id]
        prefix = [tokenizer.bos_token_id, 3] + tokenizer.encode(record["article"]) + [4]
        expected_ids = torch.tensor(prefix + summary)
        mask = batch["attention_mask"][index].bool()
        labels = batch["labels"][index]
        assert torch.equal(batch["input_ids"][index, mask], expected_ids)
        assert labels[~mask].eq(-100).all()
        assert labels[mask][:len(prefix)].eq(-100).all()
        assert torch.equal(labels[mask][len(prefix):], torch.tensor(summary))
        assert labels[-1] == tokenizer.eos_token_id
    assert batch["attention_mask"][1, 0] == 0
    assert batch["input_ids"][1, 0] == tokenizer.eos_token_id


@pytest.mark.parametrize("article_budget", [0, 1, 3])
def test_truncation_preserves_summary_and_eos(tokenizer, article_budget):
    article, summary = "alpha beta gamma delta epsilon", "one two three"
    required_length = 3 + len(tokenizer.encode(summary)) + 1  # BOS, two markers, EOS.
    encoder = extraction.TokenEncoder(tokenizer, max_length=required_length + article_budget)
    batch = encoder.encode(article, summary)
    expected = [1, 3] + tokenizer.encode(article)[:article_budget] + [4, 10, 11, 12, 2]
    assert batch["input_ids"].tolist() == [expected]
    assert batch["labels"].tolist() == [[-100] * (3 + article_budget) + [10, 11, 12, 2]]


@pytest.mark.parametrize("summary", ["one two three", ""])
def test_raises_when_summary_and_eos_cannot_fit(tokenizer, summary):
    required_length = 3 + len(tokenizer.encode(summary)) + 1
    encoder = extraction.TokenEncoder(tokenizer, max_length=required_length - 1)
    with pytest.raises(ValueError, match="full summary, EOS, BOS, and markers"):
        encoder.encode("alpha beta", summary)


def test_every_layer_matches_huggingface_forward(model, tokenizer, records):
    batch = extraction.TokenEncoder(tokenizer, max_length=32)(records)
    captured = []

    def capture_raw(_module, _inputs, output):
        raw = output[0] if isinstance(output, tuple) else output
        captured.append(raw.detach().clone())

    handles = [layer.register_forward_hook(capture_raw) for layer in model.model.layers]
    try:
        model.eval()
        with torch.inference_mode():
            expected = model(
                **batch,
                position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(0),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    model.train()  # The extractor itself must disable dropout and gradients.
    with torch.enable_grad():
        rows = extraction.forward_layer_by_layer(model, **batch)
    assert not model.training
    token_losses = []
    for index, (row, record) in enumerate(zip(rows, records)):
        summary_ids = tokenizer.encode(record["highlights"])
        target_count = len(summary_ids) + 1
        # Independent of labels: the final T inputs are the summary plus EOS.
        sources = torch.arange(batch["input_ids"].shape[1] - target_count - 1,
                               batch["input_ids"].shape[1] - 1)
        assert row["reference_target_ids"].tolist() == summary_ids + [tokenizer.eos_token_id]
        assert row["hidden_outputs"].shape == (model.config.num_hidden_layers, target_count, 16)
        assert row["distributions"].shape == (model.config.num_hidden_layers, target_count, 32)
        assert row["prediction_ids"].shape == row["matches_reference_or_final"].shape == (3, target_count)
        for tensor in row.values():
            assert tensor.device.type == "cpu" and not tensor.requires_grad

        for layer_index, raw in enumerate(captured):
            torch.testing.assert_close(row["hidden_outputs"][layer_index], raw[index, sources])
            with torch.no_grad():
                expected_probs = model.lm_head(model.model.norm(raw[index, sources])).float().softmax(-1)
            torch.testing.assert_close(row["distributions"][layer_index], expected_probs)
            assert torch.equal(row["prediction_ids"][layer_index], expected_probs.argmax(-1))

        final_probs = expected.logits[index, sources].float().softmax(-1)
        torch.testing.assert_close(row["distributions"][-1], final_probs)
        assert row["matches_reference_or_final"][-1].all()
        target_probs = row["distributions"][-1].gather(-1, row["reference_target_ids"][:, None])
        token_losses.append(-target_probs.log().flatten())
    # Hugging Face internally shifts unshifted labels; its loss is another oracle.
    torch.testing.assert_close(torch.cat(token_losses).mean(), expected.loss)


def test_padding_does_not_change_extracted_states(model, tokenizer, records):
    encoder = extraction.TokenEncoder(tokenizer, max_length=32)
    batched = extraction.forward_layer_by_layer(model, **encoder(records))
    for index in (1, 2):
        unpadded = extraction.forward_layer_by_layer(model, **encoder([records[index]]))[0]
        for name in extraction.PARQUET_SCHEMA.names:
            torch.testing.assert_close(batched[index][name], unpadded[name])


@pytest.mark.parametrize("return_tuple", [False, True])
def test_matches_reference_or_final_covers_both_branches(return_tuple):
    class FixedLayer(torch.nn.Module):
        def __init__(self, predictions):
            super().__init__()
            self.register_buffer("raw", torch.nn.functional.one_hot(
                torch.tensor([0] + predictions + [0]), num_classes=8,
            ).float().unsqueeze(0))

        def forward(self, hidden_states, attention_mask=None, position_ids=None):
            return (self.raw,) if return_tuple else self.raw

    model = torch.nn.Module()
    model.config = SimpleNamespace(vocab_size=8)
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 8)
    model.model.layers = torch.nn.ModuleList([
        FixedLayer([4, 7, 6]), FixedLayer([5, 6, 6]), FixedLayer([3, 7, 1]),
    ])
    model.model.norm = model.lm_head = torch.nn.Identity()
    row = extraction.forward_layer_by_layer(
        model, input_ids=torch.tensor([[1, 3, 4, 5, 2]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        labels=torch.tensor([[-100, -100, 4, 5, 2]]),
    )[0]
    assert row["prediction_ids"].tolist() == [[4, 7, 6], [5, 6, 6], [3, 7, 1]]
    assert row["matches_reference_or_final"].tolist() == [
        [True, True, False], [False, False, False], [True, True, True],
    ]


def test_main_writes_three_rows_and_reconstructs_losslessly(
    model, tokenizer, records, monkeypatch, tmp_path, capsys,
):
    monkeypatch.setattr("datasets.config.HF_DATASETS_CACHE", tmp_path / "datasets-cache")

    def fake_load_dataset(path, *, name, split, streaming):
        assert (path, name, split, streaming) == ("abisee/cnn_dailymail", "3.0.0", "test", True)
        # The fourth document must never be extracted.
        return IterableDataset.from_generator(lambda: iter(records + [records[0]]))

    def fake_tokenizer(path):
        assert path == "facebook/layerskip-llama2-7B"
        return tokenizer

    def fake_model(path, *, torch_dtype, attn_implementation):
        assert path == "facebook/layerskip-llama2-7B"
        assert torch_dtype == torch.float32 and attn_implementation == "sdpa"
        return model

    extracted = []
    original_forward = extraction.forward_layer_by_layer

    def capture_extracted(*args, **kwargs):
        rows = original_forward(*args, **kwargs)
        extracted.extend(rows)
        return rows

    monkeypatch.setattr(extraction, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(extraction.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(extraction.AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(extraction, "forward_layer_by_layer", capture_extracted)
    output = tmp_path / "data" / "layers.parquet"
    extraction.main(["--output", str(output), "--device", "cpu", "--batch-size", "2"])

    table = pq.read_table(output)
    assert table.num_rows == len(extracted) == 3
    assert pq.ParquetFile(output).metadata.num_row_groups == 3
    assert table.column_names == extraction.PARQUET_SCHEMA.names
    assert not output.with_suffix(".parquet.partial").exists()
    for index in range(3):
        restored = extraction.reconstruct_row(table, index)
        for name in table.column_names:
            assert torch.equal(restored[name], extracted[index][name])
            assert restored[name].dtype == extracted[index][name].dtype
        assert restored["hidden_outputs"].dtype == restored["distributions"].dtype == torch.float32
        assert restored["matches_reference_or_final"].dtype == torch.bool
    assert table.schema.field("hidden_outputs").type.value_type.value_type.value_type == pa.float32()
    printed = capsys.readouterr().out
    assert "Columns:" in printed and "Verified 3 documents" in printed
    for name in table.column_names:
        assert f"{name}: {tuple(extracted[0][name].shape)}" in printed


@pytest.mark.parametrize("document_count", [0, 2, 4])
def test_wrong_document_count_does_not_replace_output(
    model, tokenizer, records, document_count, tmp_path,
):
    selected = (records + [records[0]])[:document_count]
    batches = [extraction.TokenEncoder(tokenizer)(selected)] if selected else []
    output = tmp_path / "layers.parquet"
    output.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="Expected exactly 3 documents"):
        extraction.write_parquet(model, batches, output)
    assert output.read_bytes() == b"existing output"
    assert not output.with_suffix(".parquet.partial").exists()


@pytest.mark.parametrize("column", [
    "hidden_outputs", "prediction_ids", "matches_reference_or_final",
    "distributions", "reference_target_ids",
])
def test_verifier_rejects_misaligned_tensors(model, tokenizer, records, column, tmp_path):
    rows = extraction.forward_layer_by_layer(model, **extraction.TokenEncoder(tokenizer)(records))
    value = rows[0][column]
    rows[0][column] = value[:-1] if value.ndim == 1 else value[:, :-1]
    output = tmp_path / "misaligned.parquet"
    with pq.ParquetWriter(output, extraction.PARQUET_SCHEMA) as writer:
        for row in rows:
            writer.write_table(pa.Table.from_arrays(
                [extraction.tensor_column(row[name]) for name in extraction.PARQUET_SCHEMA.names],
                schema=extraction.PARQUET_SCHEMA,
            ))
    with pytest.raises(AssertionError):
        extraction.verify_parquet(output, tokenizer.eos_token_id)
