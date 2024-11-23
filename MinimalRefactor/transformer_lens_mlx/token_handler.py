class TokenizerHandler:
    def __init__(self, tokenizer=None):
        """
        Initialize the TokenizerHandler with a tokenizer instance.
        """
        self.tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        """
        Set the tokenizer instance.
        """
        self.tokenizer = tokenizer

    def to_tokens(self, text, padding=True):
        """
        Convert text to token IDs.

        Args:
            text (str): Input text or list of texts.
            padding (bool): Whether to pad sequences to the same length.

        Returns:
            np.ndarray: Token IDs as NumPy array.
        """
        # Ensure tokenizer has a pad_token
        if self.tokenizer.pad_token is None:
            # Set pad_token to eos_token or a custom value
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        return self.tokenizer(text, return_tensors="np", padding=padding)["input_ids"]

    def to_string(self, token_ids):
        """
        Convert token IDs to a human-readable string.
        Args:
            token_ids (np.ndarray): Token IDs as NumPy array.
        Returns:
            str: Decoded string.
        """
        return self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

    def to_str_tokens(self, token_ids):
        """
        Convert token IDs to their string representations.
        Args:
            token_ids (np.ndarray): Token IDs as NumPy array.
        Returns:
            list: List of string tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def to_single_token(self, string):
        """
        Convert a single string to its token ID.
        Args:
            string (str): Input string.
        Returns:
            int: Corresponding token ID.
        """
        return self.tokenizer.encode(string)[0]

    def to_single_str_token(self, token_id):
        """
        Convert a single token ID to its string representation.
        Args:
            token_id (int): Token ID.
        Returns:
            str: Corresponding string representation.
        """
        return self.tokenizer.decode([token_id])