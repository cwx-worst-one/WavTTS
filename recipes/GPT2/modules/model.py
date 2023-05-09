import warnings
from contextlib import contextmanager

import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel

try:
    from s3a.api import GPT2BaseModel
except Exception:
    GPT2BaseModel = None
    warnings.warn(
        "Failed to import GPT2BaseModel from s3a, please make sure you"
        " installed s3a correctly,will fallback to huggingface version."
    )


def gpt2_tiny(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_small(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_medium(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_large(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=1280,
        num_layers=36,
        num_attention_heads=20,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_xl(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=1600,
        num_layers=48,
        num_attention_heads=25,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_2B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=2048,
        num_layers=40,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_3B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=2304,
        num_layers=48,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_4B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=2304,
        num_layers=64,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_6B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=4096,
        num_layers=30,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_8B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=3072,
        num_layers=72,
        num_attention_heads=24,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_12B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=4096,
        num_layers=60,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_15B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=4096,
        num_layers=78,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_18B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=4096,
        num_layers=90,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_20B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=25,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_24B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=30,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_28B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=35,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_32B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=40,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_36B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=45,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_40B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=50,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_45B(checkpoint=True, **kwargs):
    return GPTLMModel(
        hidden_size=8192,
        num_layers=56,
        num_attention_heads=16,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt3(checkpoint=True, **kwargs):
    return GPTLMModel(
        max_seq_len=2048,
        hidden_size=12288,
        num_layers=96,
        num_attention_heads=96,
        checkpoint=checkpoint,
        **kwargs,
    )


def gpt2_pretrained(checkpoint=True):
    return GPTLMModel(pretrained_model="gpt2", checkpoint=checkpoint)


def get_gpt_model(model_name: str, checkpoint: bool = True, **kwargs) -> nn.Module:
    model_map = {
        "gpt2_tiny": gpt2_tiny,
        "gpt2_small": gpt2_small,
        "gpt2_medium": gpt2_medium,
        "gpt2_large": gpt2_large,
        "gpt2_xl": gpt2_xl,
        "gpt2_2B": gpt2_2B,
        "gpt2_3B": gpt2_3B,
        "gpt2_4B": gpt2_4B,
        "gpt2_6B": gpt2_6B,
        "gpt2_8B": gpt2_8B,
        "gpt2_12B": gpt2_12B,
        "gpt2_15B": gpt2_15B,
        "gpt2_18B": gpt2_18B,
        "gpt2_20B": gpt2_20B,
        "gpt2_24B": gpt2_24B,
        "gpt2_28B": gpt2_28B,
        "gpt2_32B": gpt2_32B,
        "gpt2_36B": gpt2_36B,
        "gpt2_40B": gpt2_40B,
        "gpt2_45B": gpt2_45B,
        "gpt3": gpt3,
        "gpt2_pretrained": gpt2_pretrained,
    }
    assert (
        model_name in model_map
    ), f"Model {model_name} not found in {model_map.keys()}"
    return model_map[model_name](checkpoint, **kwargs)


@contextmanager
def no_init_weights():
    def dummy_fn(*args):
        return

    try:
        old_init_weights = GPT2PreTrainedModel._init_weights
        GPT2PreTrainedModel._init_weights = dummy_fn
        yield
    finally:
        GPT2PreTrainedModel._init_weights = old_init_weights


class GPTLMModel(nn.Module):
    def __init__(
        self,
        model_type="GPT2LMHead",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        max_seq_len=1024,
        vocab_size=50257,
        pretrained_model=None,
        checkpoint=False,
        provider="huggingface",
    ):
        super().__init__()
        self.checkpoint = checkpoint
        if pretrained_model:
            self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        else:
            if GPT2BaseModel is not None:
                self.model = GPT2BaseModel.from_config(
                    GPT2Config(
                        n_embd=hidden_size,
                        n_layer=num_layers,
                        n_head=num_attention_heads,
                        n_positions=max_seq_len,
                        n_ctx=max_seq_len,
                        vocab_size=vocab_size,
                        activation_function="gelu",
                        fused_attn=True,
                        fused_ln=True,
                        fused_ffn=True,
                    ),
                    provider=provider,
                    model_type=model_type,
                )
            else:
                with no_init_weights():
                    self.model = GPT2LMHeadModel(
                        GPT2Config(
                            n_embd=hidden_size,
                            n_layer=num_layers,
                            n_head=num_attention_heads,
                            n_positions=max_seq_len,
                            n_ctx=max_seq_len,
                            vocab_size=vocab_size,
                        )
                    )
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        # Only return lm_logits
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=not self.checkpoint,
        )[0]


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
