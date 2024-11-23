import numpy as np
from transformer_lens_mlx.token_handler import TokenizerHandler
from transformers import AutoTokenizer

def test_to_tokens():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    handler = TokenizerHandler(tokenizer)

    text = "Hello, world!"
    tokens = handler.to_tokens(text)
    assert isinstance(tokens, np.ndarray), "Tokens should be a NumPy array"

def test_to_string():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    handler = TokenizerHandler(tokenizer)

    token_ids = [15496, 11, 995]  # Token IDs for "Hello, world"
    text = handler.to_string(token_ids)
    assert text == "Hello, world", "Token IDs should map back to the original text"

def test_to_single_token():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    handler = TokenizerHandler(tokenizer)

    token_id = handler.to_single_token("Hello")
    assert isinstance(token_id, int), "Single token should return an integer"

def test_to_single_str_token():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    handler = TokenizerHandler(tokenizer)

    token = handler.to_single_str_token(15496)
    assert token == "Hello", "Token ID should map back to the string token"