from recipes.umm2.loaders.base import BaseModelLoader
import pytorch_lightning as pl
import samantha
from mariana.utils.audio.audio_logger import AudioLogger

logger = AudioLogger()


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
        logger.info(f'Loading pretrained model from {self.ckpt_path}')

        self.modify_state_dict(state_dict)
        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict=state_dict, strict=False)
        logger.info(f"[Missing] {missing_keys}")
        logger.info(f"[Unexpected] {unexpected_keys}")
        return pl_module
    
    def modify_state_dict(self, state_dict):
        for k in list(state_dict.keys()):
            if k.startswith("model.stages.0.encoder_pre_layers"):
                k_i = int(k.split(".")[2])
                k_suffix = ".".join(k.split(".")[3:])
                k_new = f"model.stages.0.encoder_layers.{k_i}.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]
            elif k.startswith("model.stages.0.encoder_post_layers"):
                k_i = int(k.split(".")[2]) + 12
                k_suffix = ".".join(k.split(".")[3:])
                k_new = f"model.stages.0.encoder_layers.{k_i}.{k_suffix}"
                state_dict[k_new] = state_dict[k]
                print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]

class EncoderOnlyModelLoader(BaseModelLoader):
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
        map_state_dict = self.modify_state_dict_and_check_size(pl_module.state_dict(), state_dict)
        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict=map_state_dict, strict=False)
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return pl_module
    
    def modify_state_dict_and_check_size(self, current_state_dict, state_dict):
        current_state_dict_keys = current_state_dict.keys()
        map_state_dict = {}
        module_keys = ["audio_encoder", "encoder_layers", "embed_positions", "audio_transform",
                       "pos_enc"]
        
        for k in list(state_dict.keys()):
            if any([x in k for x in module_keys]):
                _k = k.replace("stages.0.", "")
                if _k not in current_state_dict_keys:
                    print(f"[ModelLoader/Missing] {_k} not in current_state_dict_keys")
                else:
                    if state_dict[k].size() != current_state_dict[_k].size():
                        print(f"[UMMModelLoader/Error] {k} size mismatch, \
                                pretrained: {state_dict[k].size()}, current: {current_state_dict[_k].size()}")
                    else:
                        map_state_dict[_k] = state_dict[k].clone()
                        print("[UMMModelLoader/Success]", k, _k)
        return map_state_dict