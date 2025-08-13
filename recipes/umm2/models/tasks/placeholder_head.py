import torch
from torch import nn
from einops import rearrange

from recipes.umm2.models.base import BaseStage

# Identity transform head used as placeholder
class Identity_Head(BaseStage):
    def __init__(
        self, 
        config, 
        takes=['latent'], 
        provides=[],
        bypasses=[],
        task="placeholder",
        loss_weight=0.0,
        lr_ratio=1.0,
        is_frozen: bool = False,
        use_conv=True,
        reduction="none"
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio, is_frozen)
        
        self.placeholder_head = nn.Identity()

    def forward(self, batch):
        latent = batch['latent']
    
        latent = self.placeholder_head(latent)

        output_dict = {} 
        return output_dict