import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

from ...requires.w2v.w2v_infer import w2v_bert_tokenization


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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
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
                w2v_tokens, input_tokens = self.prepare_feature(batch.float())
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
            print("Coarse 3AR Training, v4_3...")
        loss_offset = w2v_tokens.size(1)
        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len] - 1024
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
            + (1 + num_coarse) * 1024
        )

        input_tokens = torch.cat([w2v_tokens, eos_ids, wav_ids], dim=1)

        return w2v_tokens, input_tokens

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
