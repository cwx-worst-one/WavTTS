import torch

from .lit_coarse_3ar import CoarseModule


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
        finetune=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            finetune=finetune,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, accu = self.shared_step(batch)

        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
        return loss

    def shared_step(self, batch):
        with torch.autocast(device_type="cuda", enabled=False):
            mulan_embeds, input_tokens = self.prepare_feature(batch["audio"].float())
            org_len = input_tokens.size(1) + mulan_embeds.size(1)
            if hasattr(self.model.config, "train_len"):
                train_len = self.model.config.train_len
            else:
                train_len = org_len
            assert train_len >= org_len
            input_tokens = torch.nn.functional.pad(
                input_tokens, [0, train_len - org_len]
            )

        logits = self.model(input_ids=input_tokens, inputs_embeds=mulan_embeds)[
            "logits"
        ]

        if self.local_rank == 0:
            print("Semantic Training, V6_1...")
        loss_offset = mulan_embeds.size(1)
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[
            :, loss_offset - mulan_embeds.size(1) + 1 : org_len - mulan_embeds.size(1)
        ]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        return loss, accu

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        w2v_tokens = self.get_w2v_token(wavs)
        b, t = w2v_tokens.size()

        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]

        eos_ids = torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device) + 1024
        input_tokens = torch.cat([eos_ids, w2v_tokens], dim=1)

        return mulan_embeds, input_tokens
