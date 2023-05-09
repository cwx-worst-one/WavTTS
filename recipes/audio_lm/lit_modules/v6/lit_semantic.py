import torch

from ...requires.mulan.mulan_infer_247 import mulan_rvq_indexs
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

        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, input_tokens = self.prepare_feature(batch.float())
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
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("Semantic Training, V6...")
        loss_offset = mulan_tokens.size(1)
        loss = self.criterion(
            logits[:, loss_offset : org_len - 1, :],
            input_tokens[:, loss_offset + 1 : org_len],
        )
        accu = (
            100
            * (
                logits[:, loss_offset : org_len - 1, :].argmax(dim=-1)
                == input_tokens[:, loss_offset + 1 : org_len]
            ).sum()
            / torch.ones_like(input_tokens[:, loss_offset + 1 : org_len]).sum()
        )
        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
        return loss

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        w2v_tokens = self.get_w2v_token(wavs)
        b, t = w2v_tokens.size()

        # prepare mulan tokens
        mulan_embeds = self.get_mulan_embed(wavs)  # [b, 1, d]
        cfg_rate = self.extra_params.get("cfg_rate", 0)
        if cfg_rate > 0:
            rand_mulan = torch.nn.functional.normalize(
                torch.rand_like(mulan_embeds), dim=-1, p=2
            )
            rand_rate = torch.rand(size=[b, 1, 1], device=device)
            mulan_embeds = torch.where(rand_rate > cfg_rate, mulan_embeds, rand_mulan)

        mulan_tokens, ds = mulan_rvq_indexs(
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, t]
        # [b, 12] # offset: semantic 1024 + EOS 1
        mulan_tokens = mulan_tokens + 1024 + 1
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens
                + torch.arange(mulan_tokens.size(1)).to(device) * self.mulan_token_num
            )

        # add eos token # offset: semantic 1024
        eos_ids = torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device) + 1024
        input_tokens = torch.cat([mulan_tokens, eos_ids, w2v_tokens], dim=1)

        return mulan_tokens, input_tokens
