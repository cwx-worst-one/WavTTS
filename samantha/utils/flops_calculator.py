from functools import partial

from transformers import BertPreTrainedModel

from samantha.models.ctiga.gpt import GPTPreTrainedModel
from samantha.models.flash_llama import LlamaPreTrainedModel
from samantha.models.sparse_llama import LLaMa as SparseLLama


def llama_calculator(
    num_layers, hidden_size, intermediate_size, vocab_size, seq_len, batch_size
):
    L, H, F, T, V = num_layers, hidden_size, intermediate_size, seq_len, vocab_size
    N = 4 * H * H * L + 3 * F * H * L + V * H
    num_tokens = batch_size * seq_len
    return 3 * 2 * (N + 2 * L * H * T + 3 * L * H) * num_tokens


def bert_calculator(
    num_layers, hidden_size, intermediate_size, vocab_size, seq_len, batch_size
):
    L, H, F, T, V = num_layers, hidden_size, intermediate_size, seq_len, vocab_size
    N = 4 * H * H * L + 3 * F * H * L + V * H
    num_tokens = batch_size * seq_len
    return 3 * 2 * (N + 2 * L * H * T) * num_tokens


def retrieve_calculator(model_obj):
    if hasattr(model_obj, "flops_fn"):
        return model_obj.flops_fn

    if isinstance(model_obj, LlamaPreTrainedModel):
        return partial(
            llama_calculator,
            model_obj.config.num_hidden_layers,
            model_obj.config.hidden_size,
            model_obj.config.intermediate_size,
            model_obj.config.vocab_size,
        )
    if isinstance(model_obj, GPTPreTrainedModel):
        vocab_size = (
            getattr(model_obj.config, "num_logits", None) or model_obj.config.vocab_size
        )
        return partial(
            llama_calculator,
            model_obj.config.n_layer,
            model_obj.config.n_embd,
            model_obj.config.n_inner,
            vocab_size,
        )
    if isinstance(model_obj, SparseLLama):
        config = model_obj.params
        intermediate_size = int(config.dim * 4 * 2 / 3)
        intermediate_size = config.multiple_of * (
            (intermediate_size + config.multiple_of - 1) // config.multiple_of
        )
        return partial(
            llama_calculator,
            config.n_layers,
            config.dim,
            intermediate_size,
            config.vocab_size,
        )
    if isinstance(model_obj, BertPreTrainedModel):
        return partial(
            bert_calculator,
            model_obj.config.num_hidden_layers,
            model_obj.config.hidden_size,
            model_obj.config.intermediate_size,
            model_obj.config.vocab_size,
        )
    return None
