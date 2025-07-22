from recipes.umm2.loaders.base import BaseModelLoader
import pytorch_lightning as pl

class ModelLoader(BaseModelLoader):
    def __init__(
        self,
        ckpt_path, 
        cache_dir=None,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            cache_dir=cache_dir,
        )

    def load_model(self, pl_module: pl.LightningModule):
        state_dict = self.init_pretrained()
        print(f'Loading pretrained model from {self.ckpt_path}')

        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict=state_dict, strict=False)
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return pl_module
    
    def nn_load_model(self, model):
        state_dict = self.init_pretrained()
        print(f'Loading pretrained model from {self.ckpt_path}')

        self.modify_state_dict(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict=state_dict, strict=False)
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return model
    
    # no modification needed
    def modify_state_dict(self, state_dict):
        for k in list(state_dict.keys()):
            if k[:6] == "model.":
                k_new = k[6:]
                state_dict[k_new] = state_dict[k]
                #print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]

