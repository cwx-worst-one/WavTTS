import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler
from tqdm import tqdm

from samantha.utils.hparams import DotDict

from ...requires.w2v.w2v_infer import w2v_bert_tokenization
from ...utils.utils import sample


class CoarseModule(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.mulan_token_sep = mulan_token_sep
        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):

        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                mulan_tokens, w2v_tokens, input_tokens = self.prepare_feature(
                    batch.float()
                )
                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            # TODO: add location attention mask
            # global_step = self.trainer.global_step
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("Coarse 3AR Training, mulan247...")
        loss_offset = mulan_tokens.size(1) + w2v_tokens.size(1) + 1
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            if "ln_" in name or "bias" in name:
                print("Skip WD on {}".format(name))
                params.append({"params": [p], "weight_decay": 0.0})
            else:
                params.append({"params": [p]})
        optimizer = self.hparams.optimizer_cls(params)
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @torch.no_grad()
    def prepare_feature(self, wavs):
        device = wavs.device
        num_coarse = self.extra_params.num_coarse
        if "semantic" in self.requires.keys():
            w2v_tokens = self.get_w2v_token(wavs)
        elif "mert_model" in self.requires.keys():
            w2v_tokens = self.get_mert_token(wavs)
        else:
            raise KeyError("Invalid semantic model.")
        b, t = w2v_tokens.size()

        # offset: semantic
        wav_ids = self.get_quant(wavs)
        wav_ids = (
            wav_ids[:, :, 0:num_coarse]
            + (torch.arange(num_coarse).to(device) + 1) * 1024
        )
        wav_ids = torch.reshape(wav_ids, [b, -1])  # [b, k*t]

        # offset: semantic 1024, num_coarse
        eos_ids = (
            torch.zeros(size=[b, 1], dtype=w2v_tokens.dtype, device=device)
            + 1024
            + num_coarse * 1024
        )

        # prepare mulan embedding
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
        mulan_tokens = (
            mulan_tokens + 1024 + num_coarse * 1024 + 2
        )  # offset: semantic + coarse + EOS 2
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens + torch.arange(mulan_tokens.size(1)).to(device) * 1024
            )

        # mulan_tokens 12 + eos 1 + w2v 747 + eos 1 + ss 4800
        input_tokens = torch.cat(
            [mulan_tokens, eos_ids, w2v_tokens, eos_ids + 1, wav_ids], dim=1
        )

        return mulan_tokens, w2v_tokens, input_tokens

    @torch.no_grad()
    def get_mulan_embed(self, x):
        # [b, 1, d]
        return self.requires["mulan_infer_fn"](
            model=self.requires["mulan"],
            music=x.float()[:, 0 : 24000 * 10],
            device=x.device,
        ).unsqueeze(1)

    @torch.no_grad()
    def get_quant(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2)
        return output

    @torch.no_grad()
    def get_w2v_token(self, x):
        return w2v_bert_tokenization(
            frontend=self.requires["ssl_frontend"],
            w2v_model=self.requires["semantic"],
            wavs=x,
            centers=self.requires["centroids"],
            device=x.device,
        )

    @torch.no_grad()
    def get_mert_token(self, x):
        output_emb = self.requires["mert_model"](
            x, output_hidden_states=True
        ).hidden_states[12]
        return self.mert_tokenization(
            output_emb, self.requires["mert_centroids"], x.device
        )

    @torch.no_grad()
    def mert_tokenization(self, emb, centers, device):
        # kmeans
        b, t, d = emb.shape
        dataset = emb.view([b * t, d])
        num_points = dataset.size(0)
        # 5e8 should vary depending on the free memory on the GPU
        # Ideally, automatically ;)
        chunk_size = int(5e8)
        codes = torch.zeros(num_points, dtype=torch.long, device=device)
        centers_t = torch.transpose(centers, 0, 1)  # [1024, 1024]
        centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
        inertia = 0
        for i in range(0, num_points, chunk_size):
            begin = i
            end = min(begin + chunk_size, num_points)
            dataset_piece = dataset[begin:end, :]
            dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
            distances = torch.mm(dataset_piece, centers_t)
            distances *= -2.0
            distances += dataset_norms
            distances += centers_norms
            _, min_ind = torch.min(distances, dim=1)
            codes[begin:end] = min_ind
            inertia += distances[range(distances.shape[0]), min_ind].sum()
        codes = codes.view([b, t])
        return codes

    @torch.no_grad()
    def predict(self, mulan_tokens, semantic_samples):
        device = mulan_tokens.device
        num_coarse = self.extra_params.num_coarse
        bs = mulan_tokens.size(0)
        eos_ids = (
            torch.zeros([bs, 1], dtype=torch.long, device=device)
            + 1024
            + num_coarse * 1024
        )  # offset: w2v-bert + coarse
        mulan_tokens = (
            mulan_tokens + 1024 + num_coarse * 1024 + 2
        )  # offset: w2v-bert + coarse + EOS 2
        if self.mulan_token_sep:
            mulan_tokens = (
                mulan_tokens + torch.arange(mulan_tokens.size(1)).to(device) * 1024
            )

        slice_range = [[0, 2000]]
        # beg = 0
        # while True:
        #     end = beg + 10 * frequency * num_coarse
        #     if end >= args.duration * frequency * num_coarse:
        #         end = args.duration * frequency * num_coarse
        #         beg = end - 10 * frequency * num_coarse
        #         slice_range.append([beg, end])
        #         break
        #     else:
        #         slice_range.append([beg, end])
        #     beg += stride_in_sec * frequency * num_coarse

        prev_end = 0
        coarse_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            semantic_beg = int(cur_beg / 50 / num_coarse * 25)
            semantic_end = semantic_beg + 10 * 25
            semantic_slice = semantic_samples[:, semantic_beg:semantic_end]
            if cache_len == 0:
                input_tokens = torch.cat(
                    [mulan_tokens, eos_ids, semantic_slice, eos_ids + 1], dim=1
                )
            else:
                prefix_coarse_samples = coarse_samples[:, cur_beg : cur_beg + cache_len]
                input_tokens = torch.cat(
                    [
                        mulan_tokens,
                        eos_ids,
                        semantic_slice,
                        eos_ids + 1,
                        prefix_coarse_samples,
                    ],
                    dim=1,
                )
            past_key_values = None

            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for i in pbar:
                pbar.set_description(
                    f"Coarse [{cur_beg} - {cur_end}] [{semantic_beg} - {semantic_end}]"
                )
                coarse_outputs = self.model(
                    input_tokens, past_key_values=past_key_values, use_cache=True
                )
                logits = coarse_outputs["logits"]  # [b, t, d]
                layer_idx = i % num_coarse
                predict_logits = logits[
                    :, -1, layer_idx * 1024 : (layer_idx + 1) * 1024
                ]
                samples = sample(predict_logits, temp=0.95, mode="gumbel").unsqueeze(1)
                samples = (
                    samples + 1024 + layer_idx * 1024
                )  # [b, 1], # offset: w2v-bert
                past_key_values = coarse_outputs["past_key_values"]
                input_tokens = samples
                if coarse_samples is None:
                    coarse_samples = samples
                else:
                    coarse_samples = torch.cat([coarse_samples, samples], dim=1)
        coarse_samples = coarse_samples - 1024  # remove offset: w2v-bert
        return coarse_samples
