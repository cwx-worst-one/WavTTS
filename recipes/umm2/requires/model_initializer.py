import torch, os
from recipes.umm2.loaders.base import BaseModelLoader, local_zero_first
import sys, importlib
from recipes.umm2.modules.utils import get_task_losses
import math
import contextlib
import os
import torchaudio
import torch


class ModelLoader(BaseModelLoader):
    def __init__(
        self,
        ckpt_path,
        requires,
        cache_dir,
        device,
    ):
        super().__init__(ckpt_path, cache_dir)
        self.device = device

        # currently support {token, tag, pre-vq/post-vq latent, loss_dict, ...}
        self.requires = requires

    def load_model(self, pl_module_cls):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

        with local_zero_first():
            local_path = self.ensure_hdfs_ckpt_is_local(self.ckpt_path, self.cache_dir)
            model = pl_module_cls.load_from_checkpoint(local_path, strict=False, map_location="cpu").to(self.device).eval()

        self.pl_module = model
        return {"pl_module": model}
    
    def load_encoder_only(self, model_cls):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        return NotImplementedError
    

def init_stage3(hpath, local_rank, requires=[], cache_dir=None):
    """Init function for standard Stage3 UMM backbone."""
    from recipes.umm2.modules.stages.stage3 import Stage3RVQ

    device = torch.device(f"cuda:{local_rank}")
    loader = ModelLoader(hpath, requires=requires, cache_dir=cache_dir, device=device)
    # module_name, cls_name = pl_module_string.rsplit(".", 1)
    # module = importlib.import_module(module_name)
    # pl_module_cls = getattr(module, cls_name)
    model = loader.load_model(Stage3RVQ)
    return model["pl_module"]


def init_model(ckpt_path, pl_module_string , cache_dir="./modules/umm2/", device=None, requires=None):
    loader = ModelLoader(ckpt_path, cache_dir=cache_dir, device=device, requires=requires)
    module_name, cls_name = pl_module_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    pl_module_cls = getattr(module, cls_name)
    model = loader.load_model(pl_module_cls)
    return model["pl_module"]


def init_umm2(hpath, local_rank, pl_module_string="recipes.umm2.modules.stages.stage2.Stage2", cache_dir=None):

    model = init_model(
        ckpt_path = hpath,
        pl_module_string = pl_module_string,
        device = torch.device(f"cuda:{local_rank}")
    )
    return {"UMM2_stage2": model}
