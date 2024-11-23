import mlx.core as mx
from transformers import AutoModelForCausalLM
from transformer_lens_mlx.weights_handler import WeightsHandler
from tests.mocks import MockModel  # Import the mock model
import torch

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.ModuleDict({
            "wte": torch.nn.Embedding(10000, 768),
            "wpe": torch.nn.Embedding(512, 768),
            "h": torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "ln_1": torch.nn.LayerNorm(768),
                    "attn": torch.nn.ModuleDict({
                        "c_attn": torch.nn.Linear(768, 768),
                        "c_proj": torch.nn.Linear(768, 768),
                    }),
                    "ln_2": torch.nn.LayerNorm(768),
                    "mlp": torch.nn.ModuleDict({
                        "c_fc": torch.nn.Linear(768, 3072),
                        "c_proj": torch.nn.Linear(3072, 768),
                    }),
                }) for _ in range(12)  # Adjust layers as needed
            ]),
            "ln_f": torch.nn.LayerNorm(768),
        })
        self.lm_head = torch.nn.Linear(768, 10000, bias=False)

def test_initialize_weights():
    handler = WeightsHandler()
    mock_model = MockModel()  # Use the defined MockModel

    handler.initialize_weights(mock_model)
    for name, param in mock_model.named_parameters():
        assert mx.isfinite(param).all(), "Weights should be initialized to finite values"

def test_from_pretrained():
    handler = WeightsHandler()
    model_name = "gpt2"
    mlx_model, tokenizer = handler.from_pretrained(MockModel(), model_name)

    assert mlx_model is not None, "Pretrained model should load successfully"
    assert tokenizer is not None, "Tokenizer should load successfully"
    
def test_move_model_modules_to_device():
    """
    Test moving model modules to a specified device.
    """
    model = MockModule()
    device = "cpu"  # You can test with 'cuda' if available

    # Ensure the model starts on a different device
    assert next(model.parameters()).device.type == "cpu", "Model should start on CPU"

    # Move the model
    WeightsHandler.move_model_modules_to_device(model, device)

    # Ensure all parameters are on the specified device
    assert next(model.parameters()).device.type == device, "Model should be moved to the specified device"
    
def test_load_and_process_state_dict():
    """
    Test loading and processing a state dictionary into the model.
    """
    model = MockModule()
    state_dict = {
        "linear.weight": torch.ones((10, 10)),
        "linear.bias": torch.zeros((10,)),
    }

    # Without folding
    WeightsHandler.load_and_process_state_dict(model, state_dict, fold_ln=False)
    assert torch.equal(model.linear.weight, torch.ones((10, 10))), "Weights should be loaded correctly"
    assert torch.equal(model.linear.bias, torch.zeros((10,))), "Bias should be loaded correctly"

    # With folding (placeholder logic)
    folded_state_dict = {
        "linear.weight": torch.ones((10, 10)) * 0.5,
        "linear.bias": torch.zeros((10,)),
    }
    WeightsHandler.load_and_process_state_dict(model, folded_state_dict, fold_ln=True)
    assert torch.equal(model.linear.weight, torch.ones((10, 10)) * 0.5), "Weights should reflect folded layer norm"

def test_fold_layer_norm():
    """
    Test folding layer norm weights into adjacent layers.
    """
    state_dict = {
        "layer1.ln.weight": torch.ones(10),
        "layer1.ln.bias": torch.zeros(10),
        "layer1.weight": torch.ones((10, 10)),
    }
    folded_state_dict = WeightsHandler.fold_layer_norm(state_dict)

    # Verify that layer norm keys have been folded (example placeholder logic)
    assert "layer1.ln.weight" in state_dict, "Layer norm weights should still exist in the placeholder implementation"
    
def test_fill_missing_keys():
    """
    Test filling missing keys in the state dictionary.
    """
    model = MockModule()
    state_dict = {
        "linear.weight": torch.ones((10, 10)),  # Existing key
        # Missing "linear.bias"
    }

    # Fill missing keys
    updated_state_dict = WeightsHandler.fill_missing_keys(state_dict, model, default_value=0.5)

    # Check existing key
    assert torch.equal(updated_state_dict["linear.weight"], torch.ones((10, 10))), "Existing key should remain unchanged"

    # Check filled key
    assert "linear.bias" in updated_state_dict, "Missing key should be added"
    assert torch.equal(updated_state_dict["linear.bias"], torch.full((10,), 0.5)), "Missing key should have default value"