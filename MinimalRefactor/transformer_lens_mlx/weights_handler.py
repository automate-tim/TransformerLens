import mlx.core as mx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class WeightsHandler:
    def initialize_weights(self, model):
        """
        Initialize model weights.
        """
        for name, param in model.named_parameters():
            model.params[name] = mx.random.normal(
                shape=param.shape, 
                dtype=param.dtype, 
                loc=0.0, 
                scale=0.02
            )

    def from_pretrained(self, model, model_name, *args, **kwargs):
        """
        Load a pretrained model and adapt it for MLX compatibility.
        Args:
            model: Model instance to load weights into.
            model_name (str): Name of the pretrained model.
        Returns:
            Tuple[model, tokenizer]: Updated model and tokenizer.
        """
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, *args, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        state_dict = hf_model.state_dict()

        mlx_state_dict = {}
        for name, param in state_dict.items():
            numpy_array = param.cpu().numpy()

            # Create MLX array without explicitly specifying dtype
            mlx_state_dict[name] = mx.array(numpy_array)

        model.load_state_dict(mlx_state_dict)
        return model, tokenizer
    
    @staticmethod
    def move_model_modules_to_device(model, device):
        """
        Move model modules to the specified device.

        Args:
            model: The model instance to move.
            device (str): The target device ('cpu', 'cuda', or 'mps').

        Returns:
            None
        """
        for name, module in model.named_children():
            module.to(device)
            
    @staticmethod
    def load_and_process_state_dict(model, state_dict, fold_ln=False):
        """
        Loads a state dictionary into the model, with optional layer norm folding.

        Args:
            model: The model instance to load the state dictionary into.
            state_dict (dict): The state dictionary containing weights.
            fold_ln (bool): Whether to fold layer norm into adjacent weights.

        Returns:
            None
        """
        # Optionally fold layer norm
        if fold_ln:
            state_dict = WeightsHandler.fold_layer_norm(state_dict)

        # Load the state dictionary
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        # Print out any missing or unexpected keys
        if missing_keys:
            print(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
            
    @staticmethod
    def fold_layer_norm(state_dict):
        """
        Fold layer norm weights into adjacent layers for optimization.

        Args:
            state_dict (dict): The state dictionary containing weights.

        Returns:
            dict: The modified state dictionary with folded layer norm.
        """
        # Example implementation: Modify weights as needed
        # This is just a placeholder for actual layer norm folding logic.
        # You need to identify and fold layer norm weights correctly.
        folded_state_dict = state_dict.copy()
        for key in list(folded_state_dict.keys()):
            if "ln" in key:
                # Example: Modify or fold layer norm weights
                folded_state_dict[key] *= 0.5  # Placeholder logic
        return folded_state_dict
    
    @staticmethod
    def fill_missing_keys(state_dict, model, default_value=0.0):
        """
        Ensures the state dictionary contains all keys required by the model.

        Args:
            state_dict (dict): The state dictionary to check and fill.
            model: The model instance whose keys need to be matched.
            default_value (float): The value to use for missing keys.

        Returns:
            dict: The updated state dictionary.
        """
        updated_state_dict = state_dict.copy()
        for name, param in model.named_parameters():
            if name not in state_dict:
                print(f"Filling missing key: {name} with default value {default_value}")
                updated_state_dict[name] = torch.full_like(param, default_value)
        return updated_state_dict