import torch

from .lit_coarse import CoarseModule


class SemanticModule(CoarseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        sample_duration=30,
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

        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                input_tokens, mulan_embeds = self.prepare_feature(batch.float())

        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            inputs_embeds = self.model.wte(input_tokens)
            prompt_embeds = self.model.condition(mulan_embeds.to(batch.dtype))
            inputs_embeds = torch.cat(
                [prompt_embeds, inputs_embeds], dim=1
            ).contiguous()
            logits = self.model(inputs_embeds=inputs_embeds)["logits"]

        loss = self.criterion(logits[:, 1:-1, :], input_tokens[:, 1:], mask=None)
        self.log_dict({"tr_loss": loss.item()}, prog_bar=False, rank_zero_only=True)
        return loss

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        w2v_tokens = self.get_w2v_token(wavs)
        b, t = w2v_tokens.size()

        eos_ids = torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device) + 1024

        # add eos token
        input_tokens = torch.cat([eos_ids, w2v_tokens], dim=1)

        # prepare mulan embedding
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]

        return input_tokens, mulan_embeds
