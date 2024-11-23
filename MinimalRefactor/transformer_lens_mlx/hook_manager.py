class HookValidationError(Exception):
    """Custom exception for invalid hook additions."""
    pass

class HookManager:
    def __init__(self):
        """
        Initialize the HookManager with an empty hook registry.
        """
        self.hooks = {}

    def register_hook(self, name, hook_fn):
        """
        Register a hook for a specific layer or operation.

        Args:
            name (str): Name of the layer or tensor to attach the hook.
            hook_fn (callable): The hook function to apply.
        """
        self.hooks[name] = hook_fn

    def apply_hooks(self, layer_name, activations):
        """
        Apply registered hooks to a layer's activations.

        Args:
            layer_name (str): Name of the layer.
            activations (array): Activations from the layer.

        Returns:
            array: Modified activations after applying hooks.
        """
        if layer_name in self.hooks:
            return self.hooks[layer_name](activations)
        return activations

    def run_with_cache(self, inputs, forward_fn):
        """
        Runs the forward pass while populating a cache for hooks.

        Args:
            inputs: The input data for the model.
            forward_fn: The forward function of the model.

        Returns:
            outputs: The outputs of the forward pass.
            cache: The populated cache.
        """
        cache = {}
        outputs = forward_fn(inputs, return_cache=cache)
        return outputs, cache
    
    def check_hooks_to_add(self, 
        cfg,
        hook_point_name: str,
        error_prefix: str = "Cannot add hook",
    ) -> None:
        """
        Ensures hooks are only added to configured components.

        Args:
            cfg: Configuration object containing model setup details.
            hook_point_name: Name of the hook point.
            error_prefix: Custom error message prefix.

        Raises:
            HookValidationError: If a hook is added where the corresponding configuration is False.
        """
        if hook_point_name.endswith("attn.hook_result") and not cfg.use_attn_result:
            raise HookValidationError(f"{error_prefix} {hook_point_name} if use_attn_result is False")
        if hook_point_name.endswith(("hook_q_input", "hook_k_input", "hook_v_input")) and not cfg.use_split_qkv_input:
            raise HookValidationError(f"{error_prefix} {hook_point_name} if use_split_qkv_input is False")
        if hook_point_name.endswith("mlp_in") and not cfg.use_hook_mlp_in:
            raise HookValidationError(f"{error_prefix} {hook_point_name} if use_hook_mlp_in is False")
        if hook_point_name.endswith("attn_in") and not cfg.use_attn_in:
            raise HookValidationError(f"{error_prefix} {hook_point_name} if use_attn_in is False")