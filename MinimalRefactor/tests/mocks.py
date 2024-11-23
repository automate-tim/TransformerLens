import torch

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def load_state_dict(self, state_dict, strict=True):
        """
        Mock implementation of load_state_dict to mimic PyTorch behavior.

        Args:
            state_dict (dict): The state dictionary containing weights.
            strict (bool): Whether to enforce strict matching of keys.

        Returns:
            (list, list): Missing keys and unexpected keys.
        """
        missing_keys = []
        unexpected_keys = []

        for key in state_dict:
            if hasattr(self, key.replace(".", "_")):  # Mimic attribute lookup
                setattr(self, key.replace(".", "_"), state_dict[key])
            else:
                unexpected_keys.append(key)

        # Simulate missing keys
        for name, param in self.named_parameters():
            if name not in state_dict:
                missing_keys.append(name)

        if strict and (missing_keys or unexpected_keys):
            raise ValueError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        return missing_keys, unexpected_keys