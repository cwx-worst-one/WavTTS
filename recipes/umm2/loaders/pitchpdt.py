from recipes.umm2.loaders.base import BaseModelLoader
import pytorch_lightning as pl

class ModelLoader(BaseModelLoader):
    def __init__(
        self,
        ckpt_path,
        cache_dir=None,
        model_position=None,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            cache_dir=cache_dir,
        )
        self.model_position = model_position
        # PipelineModel: specify the order of "stages" (e.g., 0, ...) in the ModuleList
        # SpanModel: specify the order of "spans" (e.g., 0, 1, 2) in the ModuleList

    def load_model(self, pl_module: pl.LightningModule):
        self.device = pl_module.local_rank 
        # overwrite self.device with local_rank as specified in the original code
        state_dict = self.init_pretrained()
        print(f'Loading Pitch prediction model from {self.ckpt_path}')
        if self.model_position is not None:
            pl_module.model.stages[self.model_position['stages']].spans[self.model_position["spans"]].pitch_pdt.load_and_eval(state_dict)
        else:
            pl_module.model.pitchpdt.load_and_eval(state_dict)
        return pl_module
