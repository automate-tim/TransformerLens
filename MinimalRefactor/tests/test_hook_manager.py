import mlx.core as mx
import pytest
from transformer_lens_mlx.hook_manager import HookManager, HookValidationError

class MockConfig:
    def __init__(self, use_attn_result, use_split_qkv_input, use_hook_mlp_in, use_attn_in):
        self.use_attn_result = use_attn_result
        self.use_split_qkv_input = use_split_qkv_input
        self.use_hook_mlp_in = use_hook_mlp_in
        self.use_attn_in = use_attn_in

def test_register_and_apply_hooks():
    handler = HookManager()

    def hook_fn(tensor):
        return tensor * 2

    handler.register_hook("test_layer", hook_fn)
    activations = mx.array([1, 2, 3])
    modified_activations = handler.apply_hooks("test_layer", activations)

    assert (modified_activations == activations * 2).all(), "Hook function should modify activations"

def test_run_with_cache():
    handler = HookManager()

    # Update mock_forward to accept 'return_cache' as an argument
    def mock_forward(inputs, return_cache=None):
        """
        Simulates a forward pass and populates the cache.
        """
        if return_cache is not None and isinstance(return_cache, dict):
            return_cache["test"] = inputs * 2  # Populate the cache
        return inputs * 2

    # Define inputs for the test
    inputs = mx.array([1, 2, 3])

    # Call run_with_cache and verify the output
    outputs, cache = handler.run_with_cache(inputs, mock_forward)

    assert "test" in cache, "Cache should contain activations"
    assert (outputs == inputs * 2).all(), "Forward pass output should match the modified activations"
    
def test_check_hooks_to_add():
    # Valid configurations
    valid_cfg = MockConfig(True, True, True, True)
    handler = HookManager()
    try:
        handler.check_hooks_to_add(valid_cfg, "layer1.attn.hook_result")
        handler.check_hooks_to_add(valid_cfg, "layer1.hook_q_input")
        handler.check_hooks_to_add(valid_cfg, "layer1.mlp_in")
        handler.check_hooks_to_add(valid_cfg, "layer1.attn_in")
    except HookValidationError:
        pytest.fail("check_hooks_to_add raised a HookValidationError unexpectedly!")

    # Invalid configurations
    invalid_cfg = MockConfig(False, False, False, False)
    with pytest.raises(HookValidationError, match="Cannot add hook layer1.attn.hook_result if use_attn_result is False"):
        handler.check_hooks_to_add(invalid_cfg, "layer1.attn.hook_result")
    with pytest.raises(HookValidationError, match="Cannot add hook layer1.hook_q_input if use_split_qkv_input is False"):
        handler.check_hooks_to_add(invalid_cfg, "layer1.hook_q_input")
    with pytest.raises(HookValidationError, match="Cannot add hook layer1.mlp_in if use_hook_mlp_in is False"):
        handler.check_hooks_to_add(invalid_cfg, "layer1.mlp_in")
    with pytest.raises(HookValidationError, match="Cannot add hook layer1.attn_in if use_attn_in is False"):
        handler.check_hooks_to_add(invalid_cfg, "layer1.attn_in")