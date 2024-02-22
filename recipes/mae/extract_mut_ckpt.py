import torch

from recipes.mae.models.mut import MuT
from recipes.mae.modules.pl_module import LitMutMAEModule

litmodel = LitMutMAEModule.load_from_checkpoint("/mnt/bn/mm-data/projects/mulan/mae_xuchen/MuT_MAE/1212_sstk_mae_vanilla/checkpoints/mutmae-step=563200-loss_0=7-kaggle.ckpt")
torch.save(litmodel.mae.encoder.state_dict(), "/mnt/bn/mm-data/projects/mulan/mae_xuchen/MuT_MAE/1212_sstk_mae_vanilla/checkpoints/mutmae-step=563200-loss_0=7-kaggle.pth")
# mut = MuT(
#     spec_shape=(128, 1000),
#     patch_shape=(128, 2),
#     num_classes=1000,
#     sample_rate=24000,
#     dim=1280,
#     depth=32,
#     heads=16,
#     dim_head=80,
#     channels=1,
#     mlp_dim=5120,
#     checkpointing=True,
#     use_flash_attn=True,
# )
mut = MuT(
    spec_shape=(128, 1000),
    patch_shape=(128, 4), #(128, 2)
    num_classes=1000,
    sample_rate=24000,
    dim=1280,
    depth=32, #32
    heads=16,
    dim_head=80,
    channels=1,
    mlp_dim=5120,
    checkpointing=True,
    use_flash_attn=False,
)
state_dict = torch.load("/mnt/bn/mm-data/projects/mulan/mae_xuchen/MuT_MAE/1212_sstk_mae_vanilla/checkpoints/mutmae-step=563200-loss_0=7-kaggle.pth", map_location="cpu")
mut.load_state_dict(state_dict, strict=False)
