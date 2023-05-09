import torch
from tqdm import tqdm

from ...utils.utils import sample
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

        with self.profiler.profile("[LightningModule]SemanticModule.prepare_feature"):
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

        with self.profiler.profile("[LightningModule]SemanticModule.model_forward"):
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("Semantic Training, mulan247...")
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
        if "semantic" in self.requires.keys():
            w2v_tokens = self.get_w2v_token(wavs)
        elif "mert_model" in self.requires.keys():
            w2v_tokens = self.get_mert_token(wavs)
        else:
            raise KeyError("Invalid semantic model.")
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

        mulan_tokens, ds = self.requires["mulan_rvq_fn"](
            mulan_embeds.squeeze(1), self.requires["mulan_centers"]
        )  # [b, t]
        # [b, 12] # offset: semantic 1024 + EOS 1
        mulan_tokens = mulan_tokens + 1024 + 1
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens + torch.arange(mulan_tokens.size(1)).to(device) * 1024
            )

        # add eos token
        eos_ids = torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device) + 1024
        input_tokens = torch.cat([mulan_tokens, eos_ids, w2v_tokens], dim=1)

        return mulan_tokens, input_tokens

    @torch.no_grad()
    def predict(self, mulan_tokens):
        device = mulan_tokens.device
        bs = mulan_tokens.size(0)
        eos_ids = (
            torch.zeros([bs, 1], dtype=torch.long, device=device) + 1024
        )  # offset: w2v-bert
        mulan_tokens = mulan_tokens + 1024 + 1  # [b, 12] # offset: w2-vert + EOS 1
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens + torch.arange(mulan_tokens.size(1)).to(device) * 1024
            )

        slice_range = [[0, 247]]
        # beg = 0
        # while True:
        #     end = beg + 10 * semantic_frame_rate + semantic_offset
        #     if end >= args.duration * semantic_frame_rate + semantic_offset:
        #         end = args.duration * semantic_frame_rate + semantic_offset
        #         beg = end - (10 * semantic_frame_rate + semantic_offset)
        #         slice_range.append([beg, end])
        #         break
        #     else:
        #         slice_range.append([beg, end])
        #     beg += stride_in_sec * semantic_frame_rate
        prev_end = 0
        semantic_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            if cache_len == 0:
                input_tokens = torch.cat([mulan_tokens, eos_ids], dim=1)
            else:
                prefix_semantic_samples = semantic_samples[
                    :, cur_beg : cur_beg + cache_len
                ]
                input_tokens = torch.cat(
                    [mulan_tokens, eos_ids, prefix_semantic_samples], dim=1
                )
            past_key_values = None

            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for _ in pbar:
                pbar.set_description(f"Semantic [{cur_beg} - {cur_end}]")
                semantic_outputs = self.model(
                    input_tokens, past_key_values=past_key_values, use_cache=True
                )
                logits = semantic_outputs["logits"]  # [b, t, d]
                predict_logits = logits[:, -1, 0:1024]
                samples = sample(predict_logits, temp=1, mode="gumbel")
                past_key_values = semantic_outputs["past_key_values"]
                input_tokens = samples
                if semantic_samples is None:
                    semantic_samples = samples
                else:
                    semantic_samples = torch.cat([semantic_samples, samples], dim=1)
        return semantic_samples
