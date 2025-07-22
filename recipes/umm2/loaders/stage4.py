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
                del state_dict[k]
        for k in list(state_dict.keys()):
            if k[:26] == "stages.1.spans.3.outlayers":
                k_new = k.replace("stages.1.spans.3.outlayers", "stages.1.spans.0.outlayers")
                state_dict[k_new] = state_dict[k]
                #print(f"[MSD] {k} -> {k_new}")
                del state_dict[k]

class DualTokenizerModelLoader(ModelLoader):
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
        # import pdb; pdb.set_trace()
        return pl_module
  

## load models that trained on UMM branch
class UMMModelLoader(ModelLoader):
    def __init__(
        self,
        ckpt_path, 
        cache_dir=None,
        copy_encoder=True,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            cache_dir=cache_dir,
        )
        self.copy_encoder = copy_encoder

    def load_model(self, pl_module: pl.LightningModule):
        state_dict = self.init_pretrained()
        print(f'Loading pretrained model from {self.ckpt_path}')
        map_state_dict = self.modify_state_dict_and_check_size(pl_module.state_dict(), state_dict)
        if self.copy_dual_encoder:
            map_state_dict.update(self.copy_dual_encoder(pl_module.state_dict(), state_dict))
        missing_keys, unexpected_keys = pl_module.load_state_dict(state_dict=map_state_dict, strict=False)
        print(f"[Missing] {missing_keys}")
        print(f"[Unexpected] {unexpected_keys}")
        return pl_module
    
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

    def copy_dual_encoder(self, current_state_dict, pretrained_state_dict):
        map_state_dict = {}
        # copy encoder parameters to dual encoder
        for k in list(pretrained_state_dict.keys()):
            if any([x in k for x in ["audio_encoder", "encoder_layers", "embed_positions",
                                    "audio_transform", "vq_proj_in", "vq_proj_out"]]):
                _k = k.split(".")
                if "encoder_layers" in k and int(_k[_k.index("encoder_layers") + 1]) >= 12:
                    continue
                if "vq_proj_in" in k or "vq_proj_out" in k:
                    # model.vq_proj_in.weight => model.stages.2.quantization.vq_proj_in.weight
                    _k.insert(1, "stages.2.quantization")
                else:
                    _k.insert(1, "stages.2")
                new_k = ".".join(_k)
                if new_k not in current_state_dict.keys():
                    print(f"[UMMModelLoader/Copy/Missing] {k}->{new_k} not in current_state_dict_keys")
                else:
                    if pretrained_state_dict[k].size() != current_state_dict[new_k].size():
                        print(f"[UMMModelLoader/Copy/Error] {k} size mismatch, \
                                pretrained: {pretrained_state_dict[k].size()}, current: {current_state_dict[new_k].size()}")
                    else:
                        map_state_dict[new_k] = pretrained_state_dict[k]
                        print("[UMMModelLoader/Copy/Success]", k, new_k)
        return map_state_dict
