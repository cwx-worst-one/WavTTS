import torch

from .lit_coarse import CoarseModule


class FineModule(CoarseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        sample_duration=10,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        device = batch.device

        sample_rate, hop_size, sample_duration = (
            self.extra_params.sample_rate,
            self.extra_params.hop_size,
            self.hparams.sample_duration,
        )
        num_coarse, num_fine, num_res, quant_token_num = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
            self.extra_params.num_res,
            self.extra_params.quant_token_num,
        )
        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                wav_ids = self.get_quant(batch.float())

            b = wav_ids.size(0)
            wav_ids = wav_ids[:, 0 : sample_rate // hop_size * sample_duration]

            offset = torch.arange(num_coarse) * quant_token_num
            coarse_ids = (wav_ids[:, :, 0:num_coarse] + offset.to(device)).reshape(
                (b, -1)
            )

            offset = (torch.arange(num_fine) + num_coarse) * quant_token_num
            fine_ids = (
                wav_ids[:, :, num_coarse : num_coarse + num_fine] + offset.to(device)
            ).reshape((b, -1))

            eos_id = num_res * 1024
            eos_ids = torch.zeros([b, 1], dtype=wav_ids.dtype, device=device) + eos_id

            input_tokens = torch.cat([coarse_ids, eos_ids, fine_ids], dim=1)

        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            logits = self.model(input_ids=input_tokens)["logits"]

        loss_mask = torch.arange(input_tokens.shape[1]) >= (coarse_ids.shape[1] + 1)
        loss_mask = loss_mask.unsqueeze(0).repeat(b, 1)[:, 1:].to(device)
        loss = self.criterion(logits[:, 0:-1, :], input_tokens[:, 1:], mask=loss_mask)
        self.log_dict({"tr_loss": loss.item()}, prog_bar=False, rank_zero_only=True)
        return loss
