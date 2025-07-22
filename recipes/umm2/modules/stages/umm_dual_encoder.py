import torch
from recipes.umm2.modules.stages.stage2 import Stage2
from recipes.umm2.modules.utils import (
    get_nuc,
    get_task_loss_weights,
    get_task_losses,
    get_quant_rate,
)

class Stage3RVQ(Stage3):
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            config=config,
            model_cls=model_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.config = config
        
    def _shared_step(self, batch):
        output_dict = self.model(batch)

        loss_dict = {"loss": output_dict["loss"]}

        mel = output_dict["mel"]
        loss_dict["bs"] = mel.shape[0]
        loss_dict["flops"] = output_dict["flops"]
        
        if "text_ids" in output_dict:
            text_ids = output_dict["text_ids"]
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)

        # RVQ related losses and metrics
        if "loss_rvq" in output_dict and output_dict["loss_rvq"] is not None:
            # code rate & quant rate
            for r in range(output_dict["vq_ids"].shape[-1]):
                quant_rate = get_quant_rate(self,
                    output_dict["vq_ids"][...,r].long(), self.config.vq_codebook_size
                )
                loss_dict[f"aux/quant_rate{r}"] = quant_rate
                if "vq_entropy" in output_dict:
                    loss_dict[f"aux/entropy{r}"] = output_dict["vq_entropy"][..., r]
                if "ppl" in output_dict:
                    loss_dict[f"aux/ppl{r}"] = output_dict["ppl"][..., r]

                loss_dict[f"loss_rvq{r}"] = output_dict["loss_rvq"][..., r]

                if self.trainer.global_step % 100 == 0:
                    code_rate = get_nuc(output_dict["vq_ids"][..., r])
                    loss_dict[f"aux/code_rate{r}"] = code_rate

            output_dict["loss_rvq"] = output_dict["loss_rvq"].sum()
            
        loss_dict.update(get_task_loss_weights(self, prefix="aux/")) # update by adding f"aux/w_loss_{task}"
        loss_dict.update(get_task_losses(self, output_dict))    # update by adding f"loss_{task}"

        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        
        return loss_dict
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch)
        loss_dict = {k: v for k, v in loss_dict.items() if "aux/" not in k} # remove aux items
        
        # skip plotting
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        loss_dict = {k: v for k, v in loss_dict.items() if 'loss' in k}  # keep only loss related items
        self.val_outputs[dataloader_idx].append(loss_dict)

