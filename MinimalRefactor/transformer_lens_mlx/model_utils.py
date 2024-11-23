import mlx.core as mx


class ModelUtils:
    @staticmethod
    def get_token_position(token, input_tokens):
        """
        Get the position of a token in the input sequence.
        """
        for i, value in enumerate(input_tokens):
            if value == token:
                return i
        return -1

    @staticmethod
    def tokens_to_residual_directions(W_U, tokens):
        """
        Map tokens to their unembedding vectors.
        """
        return W_U[tokens]  # Use indexing directly instead of gather

    @staticmethod
    def input_to_embed(input_ids, embed, pos_embed, hooks):
        """
        Convert input token IDs into embeddings, incorporating positional embeddings.

        Args:
            input_ids (mx.array): Token IDs of shape [batch_size, seq_len].
            embed (mx.array): Token embedding matrix.
            pos_embed (mx.array): Positional embedding matrix.
            hooks (dict): Dictionary of hooks to apply at this stage.

        Returns:
            mx.array: Combined embeddings with positional information.
        """
        # Token embeddings
        token_embeddings = embed[input_ids]

        # Positional embeddings
        positional_embeddings = pos_embed[: input_ids.shape[1]]

        # Combine token and positional embeddings
        combined_embeddings = token_embeddings + positional_embeddings

        # Apply any registered hooks
        if hooks and "hook_embed" in hooks:
            combined_embeddings = hooks["hook_embed"](combined_embeddings)

        return combined_embeddings
    
    @staticmethod
    def loss_fn(logits, targets, reduction="mean"):
        """
        Computes cross-entropy loss.

        Args:
            logits (array): Predicted logits of shape [batch_size, seq_len, vocab_size].
            targets (array): Target token IDs of shape [batch_size, seq_len].
            reduction (str): Reduction method - 'mean', 'sum', or 'none'.

        Returns:
            float or array: Loss value(s).
        """
        # Shift logits and targets to align
        shifted_logits = logits[:, :-1, :]  # Ignore the last prediction
        shifted_targets = targets[:, 1:]   # Ignore the first target

        # Compute log-softmax manually
        max_logits = mx.max(shifted_logits, axis=-1, keepdims=True)
        stabilized_logits = shifted_logits - max_logits
        log_probs = stabilized_logits - mx.log(mx.sum(mx.exp(stabilized_logits), axis=-1, keepdims=True))

        # Select log probabilities for target tokens
        selected_log_probs = mx.array([
            log_probs[i, :, shifted_targets[i]]
            for i in range(shifted_targets.shape[0])
        ])
        loss = -selected_log_probs

        # Reduction
        if reduction == "mean":
            return mx.mean(loss)
        elif reduction == "sum":
            return mx.sum(loss)
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")
        
    @staticmethod
    def to_single_token(tokenizer, string):
        """
        Converts a single string to its token ID.

        Args:
            tokenizer: The tokenizer instance.
            string (str): The string to tokenize.

        Returns:
            int: The token ID corresponding to the input string.

        Raises:
            ValueError: If the string cannot be tokenized into a single token.
        """
        tokens = tokenizer(string, return_tensors="np")["input_ids"][0]
        if len(tokens) != 1:
            raise ValueError(f"String '{string}' does not map to a single token.")
        return int(tokens[0])  # Explicitly cast to Python int
    
    @staticmethod
    def to_single_str_token(tokenizer, token_id):
        """
        Converts a single token ID to its corresponding string.

        Args:
            tokenizer: The tokenizer instance.
            token_id (int): The token ID to convert.

        Returns:
            str: The string corresponding to the token ID.

        Raises:
            ValueError: If the token ID cannot be converted to a single string token.
        """
        string = tokenizer.decode([token_id])
        tokens = tokenizer(string, return_tensors="np")["input_ids"][0]
        if len(tokens) != 1 or tokens[0] != token_id:
            raise ValueError(f"Token ID '{token_id}' does not map to a single string token.")
        return string