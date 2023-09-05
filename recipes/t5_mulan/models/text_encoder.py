import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel

from recipes.mae.models.mut import RMSNorm
from recipes.t5_mulan.models.utils import LlamaEmbedsEncoder,LlamaConfig


class AvgPooling(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.text_linear = nn.Sequential(
            RMSNorm(4096),
            nn.Linear(4096, emb_dim),
            RMSNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x, mask):
        avg = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        text_output = self.text_linear(avg)
        text_embed = F.normalize(text_output, p=2, dim=-1)
        return text_embed


# TODO add transformer blocks as text encoder
class TopTextEncoder(PreTrainedModel):
    def __init__(self, config: LlamaConfig, emb_dim: int):
        super().__init__(config)
        self.text_model = LlamaEmbedsEncoder(config)
        self.text_model.gradient_checkpointing_enable()
        self.text_linear = nn.Linear(config.hidden_size, emb_dim)

    def forward(self, input_embeds, attention_mask):

        outputs = self.text_model(
            input_embeds, attention_mask=attention_mask, position_ids=None
        )
        text_embed = self.text_linear(outputs["last_hidden_state"][:, -1, :])
        text_embed = F.normalize(text_embed, p=2, dim=1)
        return text_embed



def get_text_encoder(text_encoder="avgpool", emb_dim=512, config=None):
    if text_encoder == "avgpool":
        return AvgPooling(emb_dim)
    elif text_encoder == "llama_bi_block":
        assert config is not None
        return TopTextEncoder(config, emb_dim=emb_dim)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    import torch
    def test_t5():
        model = get_text_encoder("avgpool", 512)
        inputs = torch.randn(2, 250, 4096)
        mask = torch.ones(2, 250)
        print(model(inputs, mask).shape)

    def test_llama():
        emb_dim = 512
        config = LlamaConfig(
            input_size=4096,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=4,
            num_attention_heads=16,
            hidden_act="silu",
            max_position_embeddings=201,
            use_cache=False,
            use_RoPE=False,
        )
        model = TopTextEncoder(config, emb_dim)
        inputs = torch.randn(2, 250, 4096)
        mask = torch.ones(2, 250)
        print(model(inputs, mask).shape)