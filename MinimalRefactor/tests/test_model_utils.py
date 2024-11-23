import numpy as np
import mlx.core as mx
from transformer_lens_mlx.model_utils import ModelUtils
from transformers import AutoTokenizer


def test_get_token_position():
    tokens = mx.array([10, 20, 30, 40])
    position = ModelUtils.get_token_position(20, tokens)

    assert position == 1, "Position of token 20 should be 1"

def test_tokens_to_residual_directions():
    W_U = mx.array([[1, 2], [3, 4], [5, 6]])
    tokens = mx.array([0, 2])

    residuals = ModelUtils.tokens_to_residual_directions(W_U, tokens)
    expected_residuals = mx.array([[1, 2], [5, 6]])

    assert (residuals == expected_residuals).all(), "Residual directions should match expected values"
    
def test_input_to_embed():
    """
    Test input_to_embed functionality.
    """
    # Mock inputs
    input_ids = mx.array([[1, 2, 3], [4, 5, 6]])  # Batch of 2 sequences
    embed = mx.array(np.random.rand(10, 5))  # Token embedding matrix (10 tokens, 5 dimensions)
    pos_embed = mx.array(np.random.rand(6, 5))  # Positional embedding matrix (max 6 positions, 5 dimensions)

    # Mock hooks
    def mock_hook(embeddings):
        return embeddings * 2

    hooks = {"hook_embed": mock_hook}

    # Call input_to_embed
    combined_embeddings = ModelUtils.input_to_embed(input_ids, embed, pos_embed, hooks)

    # Expected embeddings
    token_embeddings = embed[input_ids]
    positional_embeddings = pos_embed[: input_ids.shape[1]]
    expected_embeddings = (token_embeddings + positional_embeddings) * 2  # Hook modifies embeddings

    # Assertions
    assert combined_embeddings.shape == expected_embeddings.shape, "Embedding shapes should match"
    assert np.allclose(
        np.array(combined_embeddings), np.array(expected_embeddings)
    ), "Combined embeddings should match expected output"


def test_input_to_embed_no_hook():
    """
    Test input_to_embed functionality without hooks.
    """
    # Mock inputs
    input_ids = mx.array([[1, 2, 3]])
    embed = mx.array(np.random.rand(10, 5))  # Token embedding matrix (10 tokens, 5 dimensions)
    pos_embed = mx.array(np.random.rand(6, 5))  # Positional embedding matrix (max 6 positions, 5 dimensions)

    # Call input_to_embed without hooks
    combined_embeddings = ModelUtils.input_to_embed(input_ids, embed, pos_embed, hooks={})

    # Expected embeddings
    token_embeddings = embed[input_ids]
    positional_embeddings = pos_embed[: input_ids.shape[1]]
    expected_embeddings = token_embeddings + positional_embeddings

    # Assertions
    assert combined_embeddings.shape == expected_embeddings.shape, "Embedding shapes should match"
    assert np.allclose(
        np.array(combined_embeddings), np.array(expected_embeddings)
    ), "Combined embeddings should match expected output"
    
    def test_loss_fn():
        """
        Test cross-entropy loss computation.
        """
        # Mock inputs
        logits = mx.array([[[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]]])  # [batch_size, seq_len, vocab_size]
        targets = mx.array([[0, 1]])  # [batch_size, seq_len]

        # Test loss_fn
        loss = ModelUtils.loss_fn(logits, targets, reduction="mean")
        expected_loss = -np.mean([np.log(np.exp(2.0) / (np.exp(2.0) + np.exp(1.0) + np.exp(0.1))),
                                np.log(np.exp(2.0) / (np.exp(1.0) + np.exp(2.0) + np.exp(0.1)))])
        assert np.isclose(loss, expected_loss), "Loss does not match expected value"

        # Test reduction types
        loss_sum = ModelUtils.loss_fn(logits, targets, reduction="sum")
        assert np.isclose(loss_sum, expected_loss * 2), "Sum loss does not match expected value"
        
def test_to_single_token():
    """
    Test the to_single_token function.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    single_token = ModelUtils.to_single_token(tokenizer, "Hello")
    assert isinstance(single_token, int), "The result should be an integer token ID."

    # Ensure that the string maps to exactly one token
    assert single_token == tokenizer("Hello", return_tensors="np")["input_ids"][0][0], \
        "Token ID does not match expected value."


def test_to_single_str_token():
    """
    Test the to_single_str_token function.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    token_id = ModelUtils.to_single_token(tokenizer, "Hello")
    single_string = ModelUtils.to_single_str_token(tokenizer, token_id)

    # Ensure that the token ID maps back to the original string
    assert single_string.strip() == "Hello", "The string does not match the expected token."