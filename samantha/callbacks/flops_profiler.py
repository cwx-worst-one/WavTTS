import pytorch_lightning as pl
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class FlopsProfilerCallback(pl.Callback):
    """FLOPs profiler using the DeepSpeed framework.
    As documented in https://www.deepspeed.ai/tutorials/flops-profiler/

    Args:
        pl (_type_): _description_
    """

    def __init__(self, model, profile_step: int = 20) -> None:
        self.profile_step = profile_step
        self.prof = FlopsProfiler(model)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module.global_step == self.profile_step:
            self.prof.start_profile()

            if trainer.precision == "32":
                dtype = torch.float32
            elif trainer.precision == "16-mixed":
                dtype = torch.float16
            elif trainer.precision == "bf16-mixed":
                dtype = torch.bfloat16
            else:
                raise NotImplementedError()

            with torch.autocast("cuda", dtype=dtype):
                pl_module.step(batch)

            self.prof.stop_profile()

        if pl_module.global_step == self.profile_step + 10:
            flops = self.prof.get_total_flops()
            params = self.prof.get_total_params()

            pl_module.log("parameters", params)
            pl_module.log("FLOPs", flops)

            self.prof.print_model_profile(
                profile_step=self.profile_step, detailed=False
            )
            self.prof.end_profile()
