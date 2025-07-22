from recipes.umm2.modules.pl_module import Stage0
from recipes.umm2.modules.utils import (
    get_nuc,
    get_quant_rates,
)
import torch

class Stage1(Stage0):
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

    def _shared_step(self, batch):
        
        output_dict = self.model(batch)
        #print(f'#### {type(output_dict)}')
        #print(f'#### dict {output_dict}')

        mel = output_dict["mel"]
        mel_length = output_dict["mel_length"]
        mask = torch.arange(mel.size(1), device=mel.device)[None, :] < mel_length[:, None]
        mask = mask.unsqueeze(-1)  # Add feature dimension if needed
        # Calculate mean and std using the mask
        valid_mel = mel[mask.expand_as(mel)]
        mel_mean = valid_mel.mean()
        mel_std = valid_mel.std()
                    

        quant_rate = get_quant_rates(self,
            [
                output_dict["rq_target"][:, :, i]
                for i in range(self.config.rq_codebook_num)
            ],
            self.config.rq_codebook_size,
        ) / self.config.rq_codebook_num

        loss_dict = {
            "loss": output_dict["loss"],
            "accu": output_dict["accu"],
            "flops": output_dict["flops"],
            "aux/quant_rate": quant_rate,
            "aux/num_mel_frames": mel.size(0) * mel.size(1),
            "aux/mel_mean": mel_mean,
            "aux/mel_std":mel_std,
        }
        
        if self.trainer.global_step % 100 == 0:
            code_rate = get_nuc(output_dict["rq_target"].transpose(1, 2)) 
            loss_dict["aux/code_rate"] = code_rate

        return loss_dict