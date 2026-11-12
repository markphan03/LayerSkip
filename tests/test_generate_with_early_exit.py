"""Offline autoregressive checks with real, small Llama and GRU models."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import self_speculation.self_speculation_generator as generation
from train_ee_cnn import EarlyExitGRU


class QuestionTokenizer:
    eos_token_id = None
    chat_template = None

    def __call__(self, text, **kwargs):
        self.prompt = text
        return {"input_ids": torch.tensor([[1, 4, 5]])}

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        self.output_ids = ids
        return " ".join(map(str, ids))


@pytest.fixture(params=["eager", "sdpa"])
def setup(request, tmp_path):
    torch.manual_seed(19)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, eos_token_id=None,
    )
    config._attn_implementation = request.param
    model = LlamaForCausalLM(config).eval()
    classifier = EarlyExitGRU(16, 4, num_layers=2).eval()
    path = tmp_path / "exit.pt"
    torch.save({
        "model_config": classifier.config,
        "model_state_dict": classifier.state_dict(),
        "training_args": {"model_path": "test/local-llama"},
    }, path)
    return model, QuestionTokenizer(), classifier, path


@torch.inference_mode()
def reference_tokens(model, classifier, count, threshold, forced_exits=None):
    """Independent baseline: run all layers and the training GRU's full sequence."""
    ids = torch.tensor([[1, 4, 5]])
    result, exits, raw = [], [], []

    def capture(layer, args, output):
        state = output[0] if isinstance(output, tuple) else output
        raw.append(state[:, -1:, :].clone())

    hooks = [layer.register_forward_hook(capture) for layer in model.model.layers]
    try:
        for _ in range(count):
            raw.clear()
            final = model(ids, use_cache=False).logits[:, -1]
            scores = classifier(torch.stack(raw, dim=1))[0, :, 0].sigmoid()
            eligible = scores.ge(threshold)
            eligible[-1] = True
            layer = (
                eligible.long().argmax().item() if forced_exits is None
                else forced_exits[len(result)] - 1
            )
            logits = (
                final if layer == len(raw) - 1
                else model.lm_head(model.model.norm(raw[layer]))[:, 0]
            )
            token = logits.argmax(-1).item()
            result.append(token)
            exits.append(layer + 1)
            ids = torch.cat([ids, ids.new_tensor([[token]])], dim=-1)
    finally:
        for hook in hooks:
            hook.remove()
    return result, exits


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_matches_full_sequence_training_policy_and_skips_later_layers(setup, threshold):
    model, tokenizer, classifier, path = setup
    expected, exits = reference_tokens(model, classifier, 4, threshold)
    visited = []
    hooks = [
        layer.register_forward_hook(lambda layer, args, output, i=i: visited.append(i + 1))
        for i, layer in enumerate(model.model.layers)
    ]
    try:
        answer = generation.generate_with_early_exit(
            "Why?", path, model=model, tokenizer=tokenizer,
            max_new_tokens=4, threshold=threshold,
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert tokenizer.output_ids == expected
    assert answer == " ".join(map(str, expected))
    assert tokenizer.prompt == "Question: Why?\nAnswer:"
    assert visited == [layer for depth in exits for layer in range(1, depth + 1)]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_final_layer_fallback_matches_huggingface_greedy_generation(setup):
    model, tokenizer, _, path = setup
    with torch.inference_mode():
        expected = model.generate(
            torch.tensor([[1, 4, 5]]), max_new_tokens=4,
            do_sample=False, use_cache=True, pad_token_id=0,
        )[0, 3:].tolist()
    generation.generate_with_early_exit(
        "Why?", path, model=model, tokenizer=tokenizer, max_new_tokens=4, threshold=1,
    )
    assert tokenizer.output_ids == expected


def test_changing_exit_depth_preserves_context_and_resets_gru_per_token(setup, monkeypatch):
    import train_ee_cnn

    model, tokenizer, classifier, path = setup
    depths = [3, 1, 2]
    expected, _ = reference_tokens(model, classifier, 3, 0.5, forced_exits=depths)
    # At equality (sigmoid(0) == 0.5), exit immediately. Scores are supplied
    # only to choose depths; Llama states and GRU transitions remain real.
    scores = iter([-10, -10, 0, -10, 0])
    state_is_none = []
    head_hook = classifier.head.register_forward_hook(
        lambda module, args, output: torch.full_like(output, next(scores))
    )
    gru_hook = classifier.gru.register_forward_pre_hook(
        lambda module, args: state_is_none.append(args[1] is None)
    )
    monkeypatch.setattr(train_ee_cnn, "EarlyExitGRU", lambda **kwargs: classifier)
    try:
        generation.generate_with_early_exit(
            "Why?", path, model=model, tokenizer=tokenizer,
            max_new_tokens=3, threshold=0.5,
        )
    finally:
        head_hook.remove()
        gru_hook.remove()
    assert tokenizer.output_ids == expected
    assert state_is_none == [True, False, True, True, False]


@pytest.mark.parametrize("eos_source", ["tokenizer", "model"])
def test_eos_stops_immediately_without_including_eos(setup, eos_source):
    model, tokenizer, classifier, path = setup
    expected, _ = reference_tokens(model, classifier, 1, 0)
    if eos_source == "tokenizer":
        tokenizer.eos_token_id = expected[0]
    else:
        model.generation_config.eos_token_id = [expected[0], 31]
    calls = []
    hook = model.model.layers[0].register_forward_hook(lambda *args: calls.append(1))
    try:
        answer = generation.generate_with_early_exit(
            "Why?", path, model=model, tokenizer=tokenizer, max_new_tokens=4, threshold=0,
        )
    finally:
        hook.remove()
    assert answer == ""
    assert calls == [1]


def test_sampling_and_chat_template(setup):
    model, tokenizer, classifier, path = setup
    tokenizer.chat_template = "test template"

    def chat(messages, **kwargs):
        assert messages == [{"role": "user", "content": "Why?"}]
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_tensors="pt")
        return torch.tensor([[1, 4, 5]])

    tokenizer.apply_chat_template = chat
    expected, _ = reference_tokens(model, classifier, 4, 0)
    generation.generate_with_early_exit(
        "Why?", path, model=model, tokenizer=tokenizer, max_new_tokens=4,
        threshold=0, sample=True, top_k=1,
    )
    assert tokenizer.output_ids == expected


def test_default_checkpoint_and_base_model_loading(setup, monkeypatch, tmp_path):
    model, tokenizer, _, path = setup
    load = torch.load
    paths = []

    def load_checkpoint(checkpoint_path, **kwargs):
        paths.append(checkpoint_path)
        return load(path, **kwargs)

    def load_model(model_path, **kwargs):
        assert model_path == "test/local-llama"
        assert kwargs == {"torch_dtype": torch.float32, "attn_implementation": "sdpa"}
        return model

    def load_tokenizer(model_path):
        assert model_path == "test/local-llama"
        return tokenizer

    monkeypatch.setattr(torch, "load", load_checkpoint)
    monkeypatch.setattr(generation.transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(generation.transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.chdir(tmp_path)
    generation.generate_with_early_exit("Why?", max_new_tokens=2, device="cpu")
    assert paths[0].is_absolute()
    assert paths[0].as_posix().endswith("/checkpoints/ee_cnn/best_model/final.pt")
    assert len(tokenizer.output_ids) == 2


@pytest.mark.parametrize("kwargs,match", [
    ({"threshold": 1.1}, "threshold"),
    ({"max_new_tokens": -1}, "max_new_tokens"),
    ({"sample": True, "temperature": 0}, "temperature"),
    ({"top_p": -0.1}, "top_k"),
    ({"max_new_tokens": 32}, "context length"),
])
def test_invalid_settings(setup, kwargs, match):
    model, tokenizer, _, path = setup
    with pytest.raises(ValueError, match=match):
        generation.generate_with_early_exit(
            "Why?", path, model=model, tokenizer=tokenizer, **kwargs,
        )


def test_hidden_size_mismatch(setup, tmp_path):
    model, tokenizer, _, _ = setup
    classifier = EarlyExitGRU(8, 4)
    path = tmp_path / "wrong.pt"
    torch.save({"model_config": classifier.config, "model_state_dict": classifier.state_dict()}, path)
    with pytest.raises(ValueError, match="hidden size"):
        generation.generate_with_early_exit("Why?", path, model=model, tokenizer=tokenizer)


def test_zero_budget_and_empty_question_do_not_load_models():
    assert generation.generate_with_early_exit("Why?", max_new_tokens=0) == ""
    with pytest.raises(ValueError, match="nonempty"):
        generation.generate_with_early_exit(" ")
