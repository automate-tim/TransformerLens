import mlx.core as mx
from transformer_lens_mlx.weights_handler import WeightsHandler
from transformer_lens_mlx.hook_manager import HookManager
from transformer_lens_mlx.model_utils import ModelUtils
from tests.mocks import MockModel

def test_end_to_end_workflow():
    """
    Test an end-to-end workflow using the refactored TransformerLens with MLX.
    """
    # Step 1: Initialize model and load weights
    model = MockModel()
    state_dict = {
        "linear.weight": mx.full((10, 10), 1.0),
        "linear.bias": mx.full((10,), 0.0),
    }
    state_dict = WeightsHandler.fill_missing_keys(state_dict, model)
    WeightsHandler.load_and_process_state_dict(model, state_dict)

    # Step 2: Tokenize input
    input_ids = mx.array([[1, 2, 3, 4]])
    embed = mx.full((10, 5), 0.5)  # Example token embedding matrix
    pos_embed = mx.full((10, 5), 0.2)  # Example positional embedding matrix

    combined_embeddings = ModelUtils.input_to_embed(input_ids, embed, pos_embed, hooks={})

    # Step 3: Forward pass with hooks
    hook_manager = HookManager()
    hook_manager.register_hook("layer1.attn.hook_result", lambda x: x * 2)

    def mock_forward(inputs, return_cache=None):
        if isinstance(return_cache, dict):
            return_cache["mock_key"] = "mock_value"
        return inputs * 2  # Simulate forward pass

    logits, cache = hook_manager.run_with_cache(combined_embeddings, mock_forward)

    # Step 4: Compute loss
    targets = mx.array([[1, 2, 3, 4]])
    loss = ModelUtils.loss_fn(logits, targets)

    # Step 5: Assertions
    assert logits.shape == (1, 4, logits.shape[-1]), "Logits should have correct shape"
    assert cache, "Cache should not be empty after forward pass"
    assert isinstance(loss, mx.array), "Loss should be an MLX array"

    print("Integration test passed successfully!")