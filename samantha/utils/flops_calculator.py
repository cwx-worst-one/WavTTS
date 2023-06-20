from functools import partial

from s3a.providers.ctiga.models.gpt import GPTPreTrainedModel

from samantha.models.flash_llama import LlamaPreTrainedModel


def llama_calculator(num_layers, hidden_size, intermediate_size, vocab_size, seq_len):
    L, H, F, T, V = num_layers, hidden_size, intermediate_size, seq_len, vocab_size
    N = 4 * H * H * L + 3 * F * H * L + V * H
    return 3 * 2 * (N + 2 * L * H * T + 3 * L * H)


def retrieve_calculator(model_obj):
    if isinstance(model_obj, LlamaPreTrainedModel):
        return partial(
            llama_calculator,
            model_obj.config.num_hidden_layers,
            model_obj.config.hidden_size,
            model_obj.config.intermediate_size,
            model_obj.config.vocab_size,
        )
    if isinstance(model_obj, GPTPreTrainedModel):
        return partial(
            llama_calculator,
            model_obj.config.n_layer,
            model_obj.config.n_embd,
            model_obj.config.n_inner,
            model_obj.config.vocab_size,
        )
    return None
