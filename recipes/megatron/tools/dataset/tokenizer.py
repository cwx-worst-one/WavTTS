from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS_ATTRIBUTES = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
]


def train_tokenizer(
    tokenizer_and_trainer_provider,
    train_data_provider,
    test_data_provider,
    tf_special_tokens_provider,
    output_path=None,
    check_tokenizer=True,
    num_check_iters=5,
):
    tokenizer, trainer = tokenizer_and_trainer_provider()
    train_data_iter = train_data_provider()
    test_data_iter = test_data_provider()
    # Train tokenizer
    tokenizer.train_from_iterator(train_data_iter, trainer=trainer)

    # Get special tokens
    special_tokens = tf_special_tokens_provider()
    special_tokens = {
        k: v for k, v in special_tokens.items() if k in SPECIAL_TOKENS_ATTRIBUTES
    }

    # Convert to transformers tokenizer
    tf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special_tokens)

    # Save tokenizer if output path is provided
    if output_path is not None:
        tf_tokenizer.save_pretrained(output_path)

    # Check trained tokenizer
    if check_tokenizer:
        for _ in range(num_check_iters):
            test_sample = next(test_data_iter)
            print("Origin sample:", test_sample)
            tokens = tf_tokenizer.encode(test_sample)
            print("Tokenized sample:", tokens)
            decoded = tf_tokenizer.decode(tokens, skip_special_tokens=False)
            if hasattr(tokenizer.model, "continuing_subword_prefix"):
                decoded = decoded.replace(
                    f" {tokenizer.model.continuing_subword_prefix}", ""
                )
            print("Decoded tokens:", decoded)
    return tf_tokenizer
