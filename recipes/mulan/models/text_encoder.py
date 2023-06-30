import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from samantha.utils.flops_calculator import bert_calculator


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", emb_dim: int = 128):
        super(TextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.text_model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.text_model.gradient_checkpointing_enable()
        self.text_linear = nn.Linear(1024, emb_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]
        text_output = self.text_linear(last_hidden_state[:, 0, :])
        text_embed = F.normalize(text_output, p=2, dim=1)
        return text_embed
    
    def flops_fn(self, batch_size, seq_len):
        flops = 0
        # add bert flops
        bert_config = self.text_model.config
        flops += bert_calculator(
            bert_config.num_hidden_layers,
            bert_config.hidden_size,
            bert_config.intermediate_size,
            bert_config.vocab_size,
            seq_len,
            batch_size,
        )
        # add projection layer flops
        flops += 6 * batch_size * bert_config.hidden_size * self.emb_dim
        return flops


def get_text_encoder(text_encoder="bert", emb_dim=128):
    if text_encoder == "bert":
        return TextEncoder("bert-large-uncased", emb_dim)
    else:
        raise NotImplementedError
