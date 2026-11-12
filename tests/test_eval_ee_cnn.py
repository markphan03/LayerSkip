"""Offline checks for early-exit decisions, token accuracy, and checkpoint loading."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import eval_ee_cnn as evaluation
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
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.2,
        bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = request.param
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.5, 1.5, 16))
    return model


@pytest.mark.parametrize("gru_layers", [1, 2])
def test_incremental_gru_exits_match_training_and_final_predictions_match_huggingface(
    base_model, records, gru_layers,
):
    batch = training.TokenEncoder(WordTokenizer(), max_length=32)(records)
    classifier = training.EarlyExitGRU(16, 4, num_layers=gru_layers).eval()
    with torch.inference_mode():
        training_rows = training.forward_layer_by_layer(base_model, **batch)
        scores = [classifier(row["hidden_states"].unsqueeze(0))[0].sigmoid() for row in training_rows]
        # Choose a threshold between observed scores, avoiding rounding at a boundary.
        ordered = torch.cat([value[:-1].flatten() for value in scores]).unique(sorted=True)
        gap = (ordered[1:] - ordered[:-1]).argmax().item()
        threshold = ((ordered[gap] + ordered[gap + 1]) / 2).item()
        baseline_logits = base_model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=(batch["attention_mask"].cumsum(-1) - 1).clamp_min(0),
            use_cache=False,
        ).logits
        rows = evaluation.forward_layer_by_layer(
            base_model, classifier, **batch, threshold=threshold, projection_chunk_size=1,
        )
        for index, (row, training_row, probabilities) in enumerate(zip(rows, training_rows, scores)):
            eligible = probabilities.ge(threshold)
            eligible[-1] = True
            expected_layers = eligible.long().argmax(dim=0)
            raw = training_row["hidden_states"]
            selected = raw[expected_layers, torch.arange(raw.shape[1])]
            expected_early = base_model.lm_head(base_model.model.norm(selected)).argmax(-1)
            target_positions = batch["labels"][index].ne(-100).nonzero().flatten()
            expected_final = baseline_logits[index, target_positions - 1].argmax(-1)
            assert torch.equal(row["early_exit_prediction_ids"], expected_early)
            assert torch.equal(row["final_prediction_ids"], expected_final)
            assert torch.equal(row["exit_layers"], expected_layers + 1)
            assert torch.equal(row["reference_target_ids"], batch["labels"][index, target_positions])
            assert row["reference_target_ids"][-1] == WordTokenizer.eos_token_id
            assert all(value.device.type == "cpu" and not value.requires_grad for value in row.values())
    assert all(parameter.grad is None for parameter in classifier.parameters())
    assert all(parameter.grad is None for parameter in base_model.parameters())


def fixed_models(return_tuple=False):
    class FixedLayer(torch.nn.Module):
        def __init__(self, predictions):
            super().__init__()
            self.register_buffer("raw", torch.nn.functional.one_hot(
                torch.tensor([0] + predictions + [0]), num_classes=8,
            ).float().unsqueeze(0))

        def forward(self, hidden_states, attention_mask=None, position_ids=None):
            return (self.raw,) if return_tuple else self.raw

    class IdentityGRU(torch.nn.Module):
        def forward(self, x, state=None):
            return x, state

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(8, 8)
    model.model.layers = torch.nn.ModuleList([
        FixedLayer([4, 7, 6]), FixedLayer([5, 6, 2]), FixedLayer([3, 7, 1]),
    ])
    model.model.norm = model.lm_head = torch.nn.Identity()
    classifier = torch.nn.Module()
    classifier.gru = IdentityGRU()
    classifier.head = torch.nn.Linear(8, 1)
    with torch.no_grad():
        classifier.head.weight.fill_(-2)
        classifier.head.weight[0, [2, 4, 7]] = 2
        classifier.head.bias.zero_()
    batch = {
        "input_ids": torch.tensor([[1, 3, 4, 5, 2]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 4, 5, 2]]),
    }
    return model, classifier, batch


@pytest.mark.parametrize("return_tuple", [False, True])
def test_first_accepted_layer_is_kept_and_reference_tokens_do_not_choose_exits(return_tuple):
    model, classifier, batch = fixed_models(return_tuple)
    row = evaluation.forward_layer_by_layer(model, classifier, **batch)[0]
    assert row["exit_layers"].tolist() == [1, 1, 2]
    assert row["early_exit_prediction_ids"].tolist() == [4, 7, 2]
    assert row["final_prediction_ids"].tolist() == [3, 7, 1]
    assert row["reference_target_ids"].tolist() == [4, 5, 2]

    # Changing only the scoring references must not influence exit decisions.
    changed_batch = dict(batch, labels=torch.tensor([[-100, -100, 7, 7, 7]]))
    changed = evaluation.forward_layer_by_layer(model, classifier, **changed_batch)[0]
    for name in ("exit_layers", "early_exit_prediction_ids", "final_prediction_ids"):
        assert torch.equal(row[name], changed[name])


@pytest.mark.parametrize("threshold,expected_layer", [(0.0, 1), (0.5, 1), (1.0, 3)])
def test_threshold_is_inclusive_and_missing_early_exit_falls_back_to_final(threshold, expected_layer):
    model, classifier, batch = fixed_models()
    with torch.no_grad():
        classifier.head.weight.zero_()  # Every sigmoid score is exactly 0.5.
    row = evaluation.forward_layer_by_layer(model, classifier, **batch, threshold=threshold)[0]
    assert row["exit_layers"].tolist() == [expected_layer] * 3
    if expected_layer == 3:
        assert torch.equal(row["early_exit_prediction_ids"], row["final_prediction_ids"])
    else:
        assert row["early_exit_prediction_ids"].tolist() == [4, 7, 6]


def test_single_decoder_layer_always_uses_final_predictions():
    model, classifier, batch = fixed_models()
    model.model.layers = torch.nn.ModuleList([model.model.layers[-1]])
    rows = evaluation.forward_layer_by_layer(model, classifier, **batch)
    metrics = evaluation.AccuracyMetrics(num_layers=1)
    metrics.update(rows)
    result = metrics.compute()
    assert result["early_exit_rate"] == 0
    assert result["mean_exit_layer"] == 1
    assert result["agreement_with_final_layer"] == 1


def test_accuracy_is_token_weighted_and_does_not_count_wrong_final_matches_as_correct():
    model, classifier, batch = fixed_models()
    rows = evaluation.forward_layer_by_layer(model, classifier, **batch)
    # This extra document has one incorrect prediction which matches the final layer.
    rows.append({
        "reference_target_ids": torch.tensor([1]),
        "early_exit_prediction_ids": torch.tensor([0]),
        "final_prediction_ids": torch.tensor([0]),
        "exit_layers": torch.tensor([3]),
    })
    metrics = evaluation.AccuracyMetrics(num_layers=3)
    metrics.update(rows)
    result = metrics.compute()
    assert result["documents"] == 2 and result["tokens_including_eos"] == 4
    assert result["early_exit_correct"] == 2 and result["early_exit_accuracy"] == 0.5
    assert result["final_layer_correct"] == 0 and result["final_layer_accuracy"] == 0
    assert result["agreement_with_final_layer"] == 0.5
    assert result["early_exit_rate"] == 0.75
    assert result["mean_exit_layer"] == 1.75
    assert result["exit_layer_counts"] == {"1": 2, "2": 1, "3": 1}


def test_main_loads_checkpoint_settings_but_uses_test_split_and_requested_rows(
    base_model, records, monkeypatch, tmp_path, capsys,
):
    classifier = training.EarlyExitGRU(16, 4, num_layers=2)
    with torch.no_grad():
        classifier.head.weight.zero_()
        classifier.head.bias.fill_(-10)
    checkpoint_path = tmp_path / "classifier.pt"
    torch.save({
        "model_config": classifier.config,
        "model_state_dict": classifier.state_dict(),
        "training_args": {
            "dataset": training.DATASET, "model_path": training.MODEL,
            "version": "v3", "max_length": 16, "batch_size": 2,
            "projection_chunk_size": 2,
            "split": "train", "start_row": 123, "max_documents": 1,
        },
    }, checkpoint_path)

    def fake_load_dataset(path, *, name, split, streaming):
        assert (path, name, split, streaming) == (training.DATASET, "3.0.0", "test", True)
        return RecordStream([records[0]] + records + [records[0]])

    def fake_model(path, *, torch_dtype, attn_implementation):
        assert path == training.MODEL
        assert torch_dtype == torch.float32 and attn_implementation == "sdpa"
        return deepcopy(base_model)

    monkeypatch.setattr(training, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(evaluation.AutoTokenizer, "from_pretrained", lambda path: WordTokenizer())
    monkeypatch.setattr(evaluation.AutoModelForCausalLM, "from_pretrained", fake_model)
    output = tmp_path / "metrics" / "results.json"
    result = evaluation.main([
        "--checkpoint", str(checkpoint_path), "--device", "cpu",
        "--start-row", "1", "--max-documents", "3", "--output", str(output),
    ])
    assert result == json.loads(output.read_text())
    assert result["documents"] == 3 and result["tokens_including_eos"] == 7
    assert result["settings"]["max_length"] == 16
    assert result["settings"]["projection_chunk_size"] == 2
    assert result["settings"]["split"] == "test"
    assert result["early_exit_accuracy"] == result["final_layer_accuracy"]
    assert result["early_exit_rate"] == 0
    assert result["mean_exit_layer"] == 3
    assert result["computes_all_decoder_layers"] is True
    printed = capsys.readouterr().out
    assert "Dataset indices [1, 2]" in printed and "Dataset indices [3]" in printed
    assert "Early-exit next-token accuracy:" in printed
    assert "Final-layer next-token accuracy:" in printed
    assert "exit depth is not a speed measurement" in printed


def test_empty_test_selection_raises_instead_of_reporting_zero_accuracy(monkeypatch):
    model, classifier, _ = fixed_models()
    monkeypatch.setattr(evaluation, "load_records", lambda args: iter([]))
    args = SimpleNamespace(
        max_length=32, start_row=4, batch_size=2, dataset=training.DATASET, split="test",
    )
    with pytest.raises(ValueError, match="no summary tokens"):
        evaluation.evaluate(model, WordTokenizer(), classifier, args)
