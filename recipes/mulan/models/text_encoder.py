import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from samantha.utils.flops_calculator import bert_calculator


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased", emb_dim: int = 128, output_type="cls"):
        super(TextEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.output_type = output_type
        self.text_model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.text_model.gradient_checkpointing_enable()
        if pretrained_model == 'bert-large-uncased':
            input_dim = 1024
        elif pretrained_model == "bert-base-multilingual-cased":
            input_dim = 768
        elif pretrained_model == "bert-base-chinese":
            input_dim = 768
        self.text_linear = nn.Linear(input_dim, emb_dim)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]
        if self.output_type == "cls":
            text_output = last_hidden_state[:, 0, :]
        else:
            text_output = last_hidden_state
        text_output = self.text_linear(text_output)
        text_embed = F.normalize(text_output, p=2, dim=-1)
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


def get_text_encoder(text_encoder="bert", emb_dim=128, output_type="cls"):
    if text_encoder == "bert":
        return TextEncoder("bert-large-uncased", emb_dim, output_type)
    elif text_encoder == "multilingual":
        return TextEncoder("bert-base-multilingual-cased", emb_dim, output_type)
    elif text_encoder == "chinese":
        return TextEncoder("bert-base-chinese", emb_dim, output_type)
    else:
        raise NotImplementedError
