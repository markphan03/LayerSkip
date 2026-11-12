# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from self_speculation.llama_model_utils import forward, forward_early, forward_remainder, crop_past_key_values
from self_speculation.llama_model_utils import _prepare_decoder_attention_mask, _make_causal_mask, _expand_mask, top_k_top_p_filtering, decode_next_token, ForwardResult

# Test for _make_causal_mask
def test_make_causal_mask():
    dtype = torch.float32
    device = torch.device('cpu')
    input_shape = torch.Size([2, 5])  # batch size of 2 and sequence length of 5
    past_key_values_length = 3
    
    causal_mask = _make_causal_mask(input_shape, dtype, device, past_key_values_length)
    
    # Check mask shape
    assert causal_mask.shape == (2, 1, 5, 8)  # Should account for past key values
    # Check if the mask is lower triangular with appropriate shifts for past key values
    assert torch.all(causal_mask[:, :, :, 3:] <= 0)

# Test for _expand_mask
def test_expand_mask():
    mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
    dtype = torch.float32
    expanded_mask = _expand_mask(mask, dtype)
    
    # Check shape and value propagation
    assert expanded_mask.shape == (2, 1, 3, 3)
    assert torch.all(expanded_mask[0, 0, :, 1] == torch.finfo(dtype).min)

# Test for _prepare_decoder_attention_mask
def test_prepare_decoder_attention_mask():
    model = torch.nn.Embedding(10, 3)  # Dummy model with embed_tokens
    attention_mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.int32)
    input_shape = torch.Size([2, 3])
    inputs_embeds = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    past_key_values_length = 0
    
    combined_mask = _prepare_decoder_attention_mask(model, attention_mask, input_shape, inputs_embeds, past_key_values_length)
    
    # Ensure correct shape and type
    assert combined_mask.dtype == inputs_embeds.dtype
    assert combined_mask.shape == (2, 1, 3, 3)

# Test for top_k_top_p_filtering
def test_top_k_top_p_filtering():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    top_k = 2
    top_p = 0.8
    filtered_logits = top_k_top_p_filtering(logits, top_k, top_p)
    
    # Ensure that the lowest value is filtered out
    assert torch.isinf(filtered_logits[0][0])  # Checking if the first logit is set to -inf
    assert not torch.isinf(filtered_logits[0][1])  # Ensure top 2 logits are not -inf
    assert not torch.isinf(filtered_logits[0][2])



def test_decode_next_token_argmax():
    """Test argmax decoding"""
    logits = torch.tensor([[[0.1, 0.2, 0.7]]])
    next_token, _ = decode_next_token(logits)
    assert next_token.item() == 2

def test_decode_next_token_sampling():
    """Test sampling decoding"""
    logits = torch.tensor([[[0.1, 0.2, 0.7]]])
    next_token, probabilities = decode_next_token(logits, sample=True)
    assert next_token.shape == torch.Size([1, 1])
    assert probabilities.shape == torch.Size([1, 3])


@pytest.mark.parametrize("attention_implementation", ["eager", "sdpa"])
@torch.no_grad()
def test_split_forward_matches_llama_with_cached_tokens(attention_implementation):
    """Exercise drafting, verification, and rollback without downloading weights."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = attention_implementation
    model = LlamaForCausalLM(config).eval()
    tokens = torch.tensor([[1, 5, 9, 3, 7, 2, 4, 6]])
    cache = None
    # The second round starts after rolling back the last two verified tokens.
    for start, end in [(0, 6), (4, 8)]:
        expected = model(tokens[:, :end], use_cache=True)
        full = forward(model, tokens[:, start:end], cache)
        torch.testing.assert_close(full.logits, expected.logits[:, start:])

        draft = forward_early(model, tokens[:, start:end - 2], cache, 2, None)
        draft = forward_early(
            model, tokens[:, end - 2:end - 1], draft.past_key_values,
            2, draft.exit_query_cache,
        )
        verified = forward_remainder(
            model, tokens[:, start:end], draft.past_key_values,
            2, draft.exit_query_cache,
        )
        torch.testing.assert_close(verified.logits, expected.logits[:, start:])
        expected_cache = expected.past_key_values.to_legacy_cache()
        for actual_layer, expected_layer in zip(verified.past_key_values, expected_cache):
            for actual, reference in zip(actual_layer, expected_layer):
                torch.testing.assert_close(actual, reference)
        assert len(verified.past_key_values) == config.num_hidden_layers
        cache = crop_past_key_values(verified.past_key_values, end - 2)
