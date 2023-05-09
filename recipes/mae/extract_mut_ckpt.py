import torch

from recipes.mae.models.mut import MuT
from recipes.mae.modules.pl_module import LitMutMAEModule

litmodel = LitMutMAEModule.load_from_checkpoint("mutmae-step=177600-loss_1=5-sf.ckpt")
torch.save(litmodel.mae.encoder.state_dict(), "mutmae-step=177600-loss_1=5-sf.pth")
mut = MuT(
    spec_shape=(128, 1000),
    patch_shape=(128, 2),
    num_classes=1000,
    sample_rate=24000,
    dim=1280,
    depth=32,
    heads=16,
    dim_head=80,
    channels=1,
    mlp_dim=5120,
    checkpointing=True,
    use_flash_attn=True,
)
state_dict = torch.load("mutmae-step=177600-loss_1=5-sf.pth", map_location="cpu")
mut.load_state_dict(state_dict, strict=False)
