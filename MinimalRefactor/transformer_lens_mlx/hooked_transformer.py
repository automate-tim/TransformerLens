from hook_manager import HookManager
from model_utils import ModelUtils
from token_handler import TokenizerHandler
from weights_handler import WeightsHandler


class HookedTransformer:
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer_handler = TokenizerHandler(tokenizer)
        self.hook_manager = HookManager()
        self.weights_handler = WeightsHandler()
        self.utils = ModelUtils()

        # Core model components
        self.embed = self.init_embed_layer()
        self.blocks = self.init_transformer_blocks()

    def forward(self, inputs):
        """
        Core forward method for the transformer.
        """
        pass

    # Tokenizer methods
    def to_tokens(self, text, padding=True):
        return self.tokenizer_handler.to_tokens(text, padding)

    def to_string(self, token_ids):
        return self.tokenizer_handler.to_string(token_ids)

    def to_str_tokens(self, token_ids):
        return self.tokenizer_handler.to_str_tokens(token_ids)

    def to_single_token(self, string):
        return self.tokenizer_handler.to_single_token(string)

    def to_single_str_token(self, token_id):
        return self.tokenizer_handler.to_single_str_token(token_id)

    # Weights Methods
    def initialize_weights(self):
        self.weights_handler.initialize_weights(self)

    def from_pretrained(self, model_name, *args, **kwargs):
        return self.weights_handler.from_pretrained(self, model_name, *args, **kwargs)

    # Hook Methods
    def register_hook(self, name, hook_fn):
        self.hook_manager.register_hook(name, hook_fn)

    def apply_hooks(self, layer_name, activations):
        return self.hook_manager.apply_hooks(layer_name, activations)

    def run_with_cache(self, inputs):
        return self.hook_manager.run_with_cache(inputs, self.forward)

    # Utility Methods
    def get_token_position(self, token, input_tokens):
        return self.utils.get_token_position(token, input_tokens)

    def tokens_to_residual_directions(self, W_U, tokens):
        return self.utils.tokens_to_residual_directions(W_U, tokens)