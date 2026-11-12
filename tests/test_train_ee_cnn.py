"""Offline training and checkpoint tests using a tiny random Llama."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import generate_cnn_dataset as extraction
import train_ee_cnn as training


class WordTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    vocabulary = {
        "[Prompt]": 3, "[Summary]": 4, "alpha": 5, "beta": 6,
        "gamma": 7, "one": 8, "two": 9, "three": 10,
    }

    def encode(self, text, add_special_tokens=False):
        return [self.vocabulary[word] for word in text.split()]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text) for text in texts]}


class RecordStream:
    def __init__(self, rows):
        self.rows = rows

    def skip(self, count):
        return RecordStream(self.rows[count:])

    def take(self, count):
        return RecordStream(self.rows[:count])

    def __iter__(self):
        return iter(self.rows)


@pytest.fixture
def records():
    return [
        {"article": "alpha beta gamma", "highlights": "one two three"},
        {"article": "beta", "highlights": "two"},
        {"article": "gamma alpha", "highlights": ""},
    ]


@pytest.fixture
def base_model():
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.2,
        bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config)


@pytest.fixture
def offline_main(monkeypatch, base_model, records):
    def fake_load_dataset(path, *, name, split, streaming):
        assert (path, name, split, streaming) == (
            training.DATASET, "3.0.0", "train", True,
        )
        # Select rows 4, 5, 6. Earlier/later records must never be trained on.
        return RecordStream([records[0]] * 4 + records + [records[0]])

    def fake_model(path, *, torch_dtype, attn_implementation):
        assert path == training.MODEL
        assert torch_dtype == torch.float32 and attn_implementation == "sdpa"
        return deepcopy(base_model)

    monkeypatch.setattr(training, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", lambda path: WordTokenizer())
    monkeypatch.setattr(training.AutoModelForCausalLM, "from_pretrained", fake_model)

    def run(output_dir, epochs=1):
        return training.main([
            "--device", "cpu", "--start-row", "4", "--max-documents", "3",
            "--epochs", str(epochs), "--batch-size", "2", "--max-length", "32",
            "--gru-hidden-size", "4", "--gru-layers", "1", "--save-every", "1",
            "--output-dir", str(output_dir),
        ])

    return run


def read_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_only_training_features_match_extractor_and_support_backward(base_model, records):
    batch = training.TokenEncoder(WordTokenizer(), max_length=32)(records)
    expected = extraction.forward_layer_by_layer(base_model, **batch)
    rows = training.forward_layer_by_layer(base_model, **batch, projection_chunk_size=1)
    for row, reference in zip(rows, expected):
        assert set(row) == {"hidden_states", "matches_reference_or_final", "matches_reference"}
        torch.testing.assert_close(row["hidden_states"], reference["hidden_outputs"])
        assert torch.equal(
            row["matches_reference"],
            reference["prediction_ids"].eq(reference["reference_target_ids"]),
        )
        assert torch.equal(row["matches_reference_or_final"], reference["matches_reference_or_final"])
        assert row["matches_reference_or_final"][-1].all()
        assert not row["hidden_states"].requires_grad
        assert not row["hidden_states"].is_inference()

    classifier = training.EarlyExitGRU(16, 4, num_layers=1)
    x, y = training.collate_batch(rows)
    assert x.shape == (3, 3, 4, 16)
    valid = y.ne(-100)
    assert valid.sum().item() == 3 * (4 + 2 + 1)
    logits = classifier(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], y[valid])
    # Padding contributes no loss and does not affect real token sequences.
    separate_logits = torch.cat([
        classifier(row["hidden_states"].unsqueeze(0)).flatten() for row in rows
    ])
    separate_labels = torch.cat([row["matches_reference_or_final"].float().flatten() for row in rows])
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(separate_logits, separate_labels)
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert classifier.head.weight.grad.abs().sum().item() > 0
    assert all(parameter.grad is None for parameter in base_model.parameters())


def test_training_saves_weights_optimizer_and_dataset_position(offline_main, tmp_path, capsys):
    output = tmp_path / "checkpoints"
    saved = offline_main(output)
    assert saved.parent.parent == output
    assert saved.name == "final.pt"
    assert {path.name for path in saved.parent.iterdir()} == {
        "latest.pt", "epoch_0001.pt", "final.pt", "best.pt", "ee_gru.pt",
    }
    assert (saved.parent / "best.pt").read_bytes() == (saved.parent / "ee_gru.pt").read_bytes()
    checkpoint = read_checkpoint(saved)
    progress = checkpoint["progress"]
    assert progress["epoch"] == 1
    assert progress["global_step"] == 2  # Includes the incomplete final batch.
    assert progress["last_completed_dataset_index"] == 6
    assert progress["next_dataset_index"] == 4  # Start of the next epoch.
    assert checkpoint["checkpoint_path"] == str(saved)
    assert checkpoint["model_config"] == {"input_size": 16, "hidden_size": 4, "num_layers": 1}
    assert checkpoint["optimizer_state_dict"]["state"]
    assert all(value.device.type == "cpu" for value in checkpoint["model_state_dict"].values())
    restored = training.EarlyExitGRU(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["error"] is None
    assert not list(tmp_path.rglob("*.parquet"))
    printed = capsys.readouterr().out
    assert "Dataset indices [4, 5]" in printed
    assert "Dataset indices [6]" in printed
    assert printed.count("early_exit_next_token_accuracy:") == 2


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failure_stage", ["extracting_hidden_states", "training"])
def test_exception_checkpoint_resumes_without_skipping_or_repeating_updates(
    failure, failure_stage, offline_main, monkeypatch, tmp_path, capsys,
):
    reference = read_checkpoint(offline_main(tmp_path / "reference", epochs=2))
    target, attribute = (
        (training, "forward_layer_by_layer") if failure_stage == "extracting_hidden_states"
        else (training.EarlyExitGRU, "forward")
    )
    original_forward = getattr(target, attribute)
    calls = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure("injected batch failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(target, attribute, fail_second_batch)
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(failure, match="injected batch failure"):
        offline_main(interrupted_dir, epochs=2)
    exception_path = next(interrupted_dir.glob("run_*/exception.pt"))
    checkpoint = read_checkpoint(exception_path)
    progress = checkpoint["progress"]
    assert progress["current_dataset_index"] == progress["next_dataset_index"] == 6
    assert progress["current_dataset_indices"] == [6]
    assert progress["last_completed_dataset_index"] == 5
    assert progress["epoch"] == 0 and progress["global_step"] == 1
    assert progress["stage"] == failure_stage
    assert checkpoint["error"] == f"{failure.__name__}: injected batch failure"
    printed = capsys.readouterr().err
    assert "Current dataset index (zero-based): 6" in printed
    assert "Next unprocessed dataset index: 6" in printed

    monkeypatch.setattr(target, attribute, original_forward)
    saved = training.main([
        "--resume", str(exception_path), "--device", "cpu",
        "--output-dir", str(tmp_path / "resumed"),
    ])
    resumed = read_checkpoint(saved)
    assert resumed["progress"]["global_step"] == reference["progress"]["global_step"] == 4
    for name, weights in reference["model_state_dict"].items():
        torch.testing.assert_close(resumed["model_state_dict"][name], weights, rtol=0, atol=0)
    for parameter, state in reference["optimizer_state_dict"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                resumed["optimizer_state_dict"]["state"][parameter][name], value, rtol=0, atol=0,
            )


def test_partial_optimizer_update_is_saved_but_requires_a_completed_checkpoint_to_resume(
    offline_main, monkeypatch, tmp_path, capsys,
):
    original_step = torch.optim.Adam.step
    calls = 0

    def partial_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Simulate an optimizer failing after it has already changed a weight.
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].add_(1)
            raise RuntimeError("optimizer interrupted")
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", partial_step)
    with pytest.raises(RuntimeError, match="optimizer interrupted"):
        offline_main(tmp_path)
    exception_path = next(tmp_path.glob("run_*/exception.pt"))
    checkpoint = read_checkpoint(exception_path)
    latest = read_checkpoint(exception_path.parent / "latest.pt")
    assert checkpoint["progress"]["stage"] == "optimizer_step"
    assert checkpoint["progress"]["next_dataset_index"] == 6
    name = "gru.weight_ih_l0"
    torch.testing.assert_close(
        checkpoint["model_state_dict"][name], latest["model_state_dict"][name] + 1,
    )
    with pytest.raises(SystemExit):
        training.main(["--resume", str(exception_path), "--device", "cpu"])
    assert "partial optimizer update" in capsys.readouterr().err


def test_stream_failure_reports_exact_row_and_retries_whole_batch(
    offline_main, monkeypatch, records, tmp_path, capsys,
):
    def broken_records(args, start_row=None):
        assert start_row == 4
        yield records[0]
        raise OSError("stream disconnected")

    monkeypatch.setattr(training, "load_records", broken_records)
    with pytest.raises(OSError, match="stream disconnected"):
        offline_main(tmp_path)
    checkpoint = read_checkpoint(next(tmp_path.glob("run_*/exception.pt")))
    progress = checkpoint["progress"]
    assert progress["current_dataset_index"] == 5
    assert progress["current_dataset_indices"] == [4]
    assert progress["next_dataset_index"] == 4
    assert progress["global_step"] == 0
    assert "Current dataset index (zero-based): 5" in capsys.readouterr().err


def test_failed_save_preserves_previous_checkpoint(monkeypatch, tmp_path):
    classifier = training.EarlyExitGRU(4, 2, num_layers=1)
    optimizer = torch.optim.Adam(classifier.parameters())
    output = tmp_path / "latest.pt"
    output.write_bytes(b"previous checkpoint")

    def broken_save(checkpoint, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk full"):
        training.save_checkpoint(
            output, classifier, optimizer, training.TrainingProgress(), SimpleNamespace(),
        )
    assert output.read_bytes() == b"previous checkpoint"
    assert not output.with_suffix(".pt.tmp").exists()


def test_batch_accuracy_uses_first_exit_reference_only_and_excludes_padding():
    rows = [
        {"matches_reference": torch.tensor([
            [True, False, True], [False, True, False], [False, False, True],
        ])},
        {"matches_reference": torch.tensor([[False], [True], [False]])},
    ]
    logits = torch.tensor([
        [[0., -1., -1.], [1., 1., -1.], [1., 1., -1.]],
        [[-1., 100., 100.], [-1., 100., 100.], [-1., 100., 100.]],
    ])
    # First document exits at layers 1, 2, 3; second falls back and is wrong.
    assert training.early_exit_next_token_accuracy(logits, rows) == 0.75


def test_best_checkpoint_tracks_lowest_loss_and_is_copied_at_completion(
    offline_main, monkeypatch, tmp_path,
):
    original_loss = training.nn.BCEWithLogitsLoss
    losses = iter([0.7, 0.3, 0.8, 0.6])

    class ControlledLoss(original_loss):
        def forward(self, logits, targets):
            loss = super().forward(logits, targets)
            return (loss - loss.detach()) + next(losses)

    monkeypatch.setattr(training.nn, "BCEWithLogitsLoss", ControlledLoss)
    saved = offline_main(tmp_path, epochs=2)
    best = read_checkpoint(saved.parent / "best.pt")
    final = read_checkpoint(saved)
    assert best["progress"]["global_step"] == 2
    assert best["progress"]["last_loss"] == pytest.approx(0.3)
    assert final["progress"]["best_loss"] == pytest.approx(0.3)
    assert final["progress"]["last_loss"] == pytest.approx(0.6)
    epoch_one = read_checkpoint(saved.parent / "epoch_0001.pt")
    for name, weights in best["model_state_dict"].items():
        torch.testing.assert_close(weights, epoch_one["model_state_dict"][name], rtol=0, atol=0)
    assert (saved.parent / "ee_gru.pt").read_bytes() == (saved.parent / "best.pt").read_bytes()


def test_repeated_and_resumed_runs_preserve_all_previous_files(offline_main, tmp_path):
    first = offline_main(tmp_path)
    original_files = {path: path.read_bytes() for path in first.parent.iterdir()}
    second = offline_main(tmp_path)
    assert first.parent != second.parent
    resumed = training.main([
        "--resume", str(first), "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    assert resumed.parent not in (first.parent, second.parent)
    assert (resumed.parent / "ee_gru.pt").exists()
    for path, content in original_files.items():
        assert path.read_bytes() == content
