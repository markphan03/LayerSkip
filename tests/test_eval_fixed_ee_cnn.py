"""Offline checks of fixed-layer predictions, stopping depth, and accuracy."""

from copy import deepcopy
import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import eval_fixed_ee_cnn as evaluation
import train_ee_cnn as training
from tests.test_train_ee_cnn import RecordStream, WordTokenizer


@pytest.fixture
def records():
    return [
        {"article": "alpha beta gamma", "highlights": "one two three"},
        {"article": "beta", "highlights": "two"},
        {"article": "gamma alpha", "highlights": ""},
    ]


@pytest.fixture(params=["eager", "sdpa"])
def base_model(request):
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.2,
        bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = request.param
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.5, 1.5, 16))
    return model


@pytest.mark.parametrize("exit_layer", [1, 4, 6])
def test_predictions_match_requested_layer_and_later_layers_are_never_executed(
    base_model, records, exit_layer,
):
    batch = training.TokenEncoder(WordTokenizer(), max_length=32)(records)
    captured = []

    def capture(_module, _inputs, output):
        captured.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    handles = [layer.register_forward_hook(capture) for layer in base_model.model.layers]
    try:
        base_model.eval()
        with torch.inference_mode():
            baseline = base_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(0),
                use_cache=False,
            ).logits
    finally:
        for handle in handles:
            handle.remove()

    executed = []
    handles = [
        layer.register_forward_hook(lambda _m, _i, _o, index=index: executed.append(index + 1))
        for index, layer in enumerate(base_model.model.layers)
    ]
    try:
        options = {} if exit_layer == 4 else {"exit_layer": exit_layer}
        rows = evaluation.forward_layer_by_layer(base_model, **batch, **options)
    finally:
        for handle in handles:
            handle.remove()
    assert executed == list(range(1, exit_layer + 1))
    assert sum(row["reference_target_ids"].numel() for row in rows) == 7

    with torch.inference_mode():
        for index, row in enumerate(rows):
            targets = batch["labels"][index].ne(-100).nonzero().flatten()
            raw = captured[exit_layer - 1][index, targets - 1]
            expected = base_model.lm_head(base_model.model.norm(raw)).argmax(-1)
            assert torch.equal(row["early_exit_prediction_ids"], expected)
            assert torch.equal(row["reference_target_ids"], batch["labels"][index, targets])
            assert row["reference_target_ids"][-1] == WordTokenizer.eos_token_id
            assert row["exit_layers"].tolist() == [exit_layer] * len(targets)
            if exit_layer == 6:
                assert torch.equal(expected, baseline[index, targets - 1].argmax(-1))


@pytest.mark.parametrize("return_tuple", [False, True])
def test_fixed_fourth_layer_accuracy_counts_reference_matches_including_eos(return_tuple):
    class FixedLayer(torch.nn.Module):
        def __init__(self, predictions):
            super().__init__()
            self.register_buffer("raw", torch.nn.functional.one_hot(
                torch.tensor([0] + predictions + [0]), num_classes=8,
            ).float().unsqueeze(0))

        def forward(self, hidden_states, attention_mask=None, position_ids=None):
            return (self.raw,) if return_tuple else self.raw

    class MustNotRun(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            raise AssertionError("A decoder block after layer 4 was executed.")

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 8)
    model.model.layers = torch.nn.ModuleList([
        FixedLayer([0, 0, 0]), FixedLayer([0, 0, 0]), FixedLayer([0, 0, 0]),
        FixedLayer([4, 7, 2]), MustNotRun(),
    ])
    model.model.norm = model.lm_head = torch.nn.Identity()
    rows = evaluation.forward_layer_by_layer(
        model, input_ids=torch.tensor([[1, 3, 4, 5, 2]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        labels=torch.tensor([[-100, -100, 4, 5, 2]]),
    )
    assert rows[0]["early_exit_prediction_ids"].tolist() == [4, 7, 2]
    metrics = evaluation.AccuracyMetrics()
    metrics.update(rows)
    assert metrics.compute()["early_exit_accuracy"] == pytest.approx(2 / 3)

    # A second, shorter document contributes one token, not half the total weight.
    metrics.update([{
        "reference_target_ids": torch.tensor([1]),
        "early_exit_prediction_ids": torch.tensor([0]),
    }])
    assert metrics.compute() == {
        "documents": 2, "tokens_including_eos": 4,
        "early_exit_correct": 2, "early_exit_accuracy": 0.5,
    }


@pytest.mark.parametrize("exit_layer", [0, 7])
def test_invalid_layer_is_rejected(base_model, records, exit_layer):
    batch = training.TokenEncoder(WordTokenizer(), max_length=32)(records)
    with pytest.raises(ValueError, match="between 1 and 6"):
        evaluation.forward_layer_by_layer(base_model, **batch, exit_layer=exit_layer)


def test_main_uses_checkpoint_settings_without_classifier_weights(
    base_model, records, monkeypatch, tmp_path, capsys,
):
    checkpoint_path = tmp_path / "settings_only.pt"
    # Deliberately omit classifier weights/config: fixed exits do not use them.
    torch.save({"training_args": {
        "dataset": training.DATASET, "model_path": training.MODEL,
        "version": "v3", "max_length": 16, "batch_size": 2,
        "projection_chunk_size": 2,
        "split": "train", "start_row": 123, "max_documents": 1,
    }}, checkpoint_path)

    def fake_load_dataset(path, *, name, split, streaming):
        assert (path, name, split, streaming) == (training.DATASET, "3.0.0", "test", True)
        return RecordStream([records[0]] + records + [records[0]])

    monkeypatch.setattr(training, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(evaluation.AutoTokenizer, "from_pretrained", lambda path: WordTokenizer())
    monkeypatch.setattr(
        evaluation.AutoModelForCausalLM, "from_pretrained", lambda path, **kwargs: deepcopy(base_model),
    )
    output = tmp_path / "metrics" / "layer4.json"
    result = evaluation.main([
        "--checkpoint", str(checkpoint_path), "--device", "cpu",
        "--start-row", "1", "--max-documents", "3", "--output", str(output),
    ])
    assert result == json.loads(output.read_text())
    assert result["documents"] == 3 and result["tokens_including_eos"] == 7
    assert result["settings"]["max_length"] == 16
    assert result["settings"]["projection_chunk_size"] == 2
    assert result["settings"]["split"] == "test"
    assert result["exit_layer"] == 4 and result["num_decoder_layers"] == 6
    assert result["computes_all_decoder_layers"] is False
    assert 0 <= result["early_exit_accuracy"] <= 1
    printed = capsys.readouterr().out
    assert "Dataset indices [1, 2]" in printed and "Dataset indices [3]" in printed
    assert "Layer 4 early-exit next-token accuracy:" in printed


def test_empty_selection_does_not_report_zero_accuracy():
    with pytest.raises(ValueError, match="no summary tokens"):
        evaluation.AccuracyMetrics().compute()
