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
    
    # no modification needed
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


## load models that trained on UMM branch
class UMMModelLoader(ModelLoader):
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
        model_state_dict_keys = pl_module.state_dict().keys()
        # map_state_dict = self.modify_state_dict(model_state_dict_keys, state_dict)
        # new_state_dict = self.check_size(pl_module.state_dict(), map_state_dict)
        new_state_dict = self.modify_state_dict_and_check_size(pl_module.state_dict(), state_dict)
        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict=new_state_dict, strict=False)
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return pl_module
    
    def check_size(self, current_model_state_dict, pretrained_model_state_dict):
        new_state_dict = {}
        for k, v in zip(current_model_state_dict.keys(), pretrained_model_state_dict.values()):
            if v.size() == current_model_state_dict[k].size():
                new_state_dict[k] = v
            else:
                new_state_dict[k] = current_model_state_dict[k]
                print(f"[UMMModelLoader] {k} size mismatch, pretrained: {v.size()}, current: {current_model_state_dict[k].size()}")
        return new_state_dict

    def modify_state_dict_and_check_size(self, current_state_dict, state_dict):
        current_state_dict_keys = current_state_dict.keys()
        map_state_dict = {}
        module_keys = ["audio_encoder", "encoder_layers", "embed_positions", "audio_transform", 
                       "mel_head", "ctc_head", "chroma_transform", "chroma_head", "f0_vuv_head", "vq"]
        
        for k in list(state_dict.keys()):
            if any([x in k for x in module_keys]):
                _k = k.split(".")
                if any([x in k for x in ["audio_encoder", "encoder_layers", "embed_positions",
                                        "audio_transform",]]):
                    _k.insert(1, "stages.0")
                elif any([x in k for x in ["mel_head", "ctc_head", "chroma_transform", "chroma_transform",
                                        "chroma_head", "f0_vuv_head"]]):
                    if "mel_head" in k:
                        # model.stages.1.spans.0.mel_head.conv.0.weight'
                        # model.mel_head.conv.0.weight
                        _k.insert(1, "stages.1.spans.0")
                    elif "ctc_head" in k:
                        # model.stages.1.spans.1.ctc_head.weight
                        # model.ctc_head.weight
                        _k.insert(1, "stages.1.spans.1")
                    elif "chroma_transform" in k or "chroma_head" in k:
                        _k.insert(1, "stages.1.spans.2")
                    elif "f0_vuv_head" in k:
                        _k.insert(1, "stages.1.spans.3")
                elif "vq" in k:
                    # load vq weight
                    # 'model.vq.embedding.weight', 'model.vq.embedding.cluster_size', 'model.vq.embedding.embed_avg', 
                    # 'model.vq_proj_in.weight', 'model.vq_proj_out.weight'
                    # => 
                    # model.stages.0.insert_modules.0.vq.embedding.weight', model.stages.0.insert_modules.0.vq.embedding.cluster_size'
                    # model.stages.0.insert_modules.0.vq.embedding.embed_avg
                    # model.stages.0.insert_modules.0.vq_proj_in.weight, model.stages.0.insert_modules.0.vq_proj_out.weight
                    _k.insert(1, "stages.0.insert_modules.0")

                new_k = ".".join(_k)
                if new_k not in current_state_dict_keys:
                    print(f"[UMMModelLoader/Missing] {k}->{new_k} not in current_state_dict_keys")
                else:
                    if state_dict[k].size() != current_state_dict[new_k].size():
                        print(f"[UMMModelLoader/Error] {k} size mismatch, \
                                pretrained: {state_dict[k].size()}, current: {current_state_dict[new_k].size()}")
                    else:
                        map_state_dict[new_k] = state_dict[k]
                        print("[UMMModelLoader/Success]", k, new_k)

        return map_state_dict
     
    def modify_state_dict_bak(self, map_state_dict_keys, state_dict):
        map_state_dict = {}
        
        for k in list(state_dict.keys()):
            if any([x in k for x in ["audio_encoder","encoder_layers","embed_positions",
                                     "audio_transform",]]):
                _k = k.split(".")
                _k.insert(1, "stages.0")
                stage0_new_k = ".".join(_k)
                if stage0_new_k in map_state_dict_keys:
                    map_state_dict[stage0_new_k] = state_dict[k]
                else:
                    print(f"[MSD] {stage0_new_k} not in map_state_dict_keys")

            elif "mel_head" in k:
                # model.stages.1.spans.0.mel_head.conv.0.weight'
                # model.mel_head.conv.0.weight
                _k = k.split(".")
                _k.insert(1, "stages.1.spans.0")
                stage1_span0_new_k = ".".join(_k)
                map_state_dict[stage1_span0_new_k] = state_dict[k]
            elif "ctc_head" in k:
                # model.stages.1.spans.1.ctc_head.weight
                # model.ctc_head.weight
                _k = k.split(".")
                _k.insert(1, "stages.1.spans.1")
                stage1_span1_new_k = ".".join(_k)
                map_state_dict[stage1_span1_new_k] = state_dict[k]
            elif "chroma_transform" in k or "chroma_head" in k:
                _k = k.split(".")
                _k.insert(1, "stages.1.spans.2")
                stage1_span2_new_k = ".".join(_k)
                map_state_dict[stage1_span2_new_k] = state_dict[k]
            elif "f0_vuv_head" in k:
                _k = k.split(".")
                _k.insert(1, "stages.1.spans.3")
                stage1_span3_new_k = ".".join(_k)
                map_state_dict[stage1_span3_new_k] = state_dict[k]

        return map_state_dict