import torch

from .lit_coarse_3ar import CoarseModule


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
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        device = batch.device

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
                # get coarse ids
                offset = torch.arange(num_coarse) * quant_token_num
                coarse_ids = (wav_ids[:, :, 0:num_coarse] + offset.to(device)).reshape(
                    (b, -1)
                )
                # get fine ids
                offset = (torch.arange(num_fine) + num_coarse) * quant_token_num
                fine_ids = (
                    wav_ids[:, :, num_coarse : num_coarse + num_fine]
                    + offset.to(device)
                ).reshape((b, -1))
                # get eos id
                eos_id = num_res * self.codec_token_num
                eos_ids = (
                    torch.zeros([b, 1], dtype=wav_ids.dtype, device=device) + eos_id
                )
                # final input tokens
                input_tokens = torch.cat([coarse_ids, eos_ids, fine_ids], dim=1)

                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            logits = self.model(input_ids=input_tokens)["logits"]

        if self.local_rank == 0:
            print("Fine Training, V5...")
        loss_offset = coarse_ids.size(1)

        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
        return loss
