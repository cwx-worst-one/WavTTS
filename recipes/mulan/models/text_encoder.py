import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="bert-base-uncased"):
        super(TextEncoder, self).__init__()
        self.text_model = AutoModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.text_model.gradient_checkpointing_enable()
        self.text_linear = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.text_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state = outputs["last_hidden_state"]
        text_output = self.text_linear(last_hidden_state[:, 0, :])
        text_embed = F.normalize(text_output, p=2, dim=1)
        return text_embed


def get_text_encoder(text_encoder="bert"):
    if text_encoder == "bert":
        return TextEncoder("bert-base-uncased")
    else:
        raise NotImplementedError
