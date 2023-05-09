import pytorch_lightning as pl
import torch
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict

# from ...requires.w2v.w2v_infer import w2v_bert_tokenization
# from ...requires.bshall_hubert.bshall_hubert_infer import bshall_hubert_tokenization
from ...requires.hubert.hubert_infer import hubert_tokenization


class SemanticModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        use_offline_token=False,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.criterion = criterion_cls()
        self.extra_params = DotDict(extra_params)
        self.requires = {}
        self.use_offline_token = use_offline_token
        if checkpointing:
            self.model.gradient_checkpointing_enable()

    def setup(self, stage: str) -> None:
        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath,
                   initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath,
                                             local_rank=self.local_rank))

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):

        with self.profiler.profile(
                "[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                if self.use_offline_token:
                    input_tokens, token_lens = batch
                    input_tokens = input_tokens.long()
                    token_lens = token_lens.long()
                    org_len = input_tokens.size(1)
                else:
                    wavs, wav_lens = batch
                    input_tokens = self.prepare_feature(
                        wavs.float(), wav_lens.long())
                    org_len = input_tokens.size(1)
                # if hasattr(self.model.config, "train_len"):
                #     train_len = self.model.config.train_len
                # else:
                #     train_len = org_len
                # assert train_len >= org_len
                # input_tokens = torch.nn.functional.pad(
                #     input_tokens, [0, train_len - org_len])
            
        mask = self.get_mask_from_lengths(token_lens, org_len)

        with self.profiler.profile(
                "[LightningModule]FineModule.model_forward"):
            logits = self.model(input_tokens)["logits"]

        if self.local_rank == 0:
            print("Semantic Training, V4...")
        loss_offset = 0 # no offset
        loss = self.criterion(
            logits[:, loss_offset:org_len - 1, :],
            input_tokens[:, loss_offset + 1:org_len],
            mask[:, loss_offset + 1:org_len]
        )
        accu = (
            100 * ((logits[:, loss_offset:org_len - 1, :].argmax(dim=-1)
                   == input_tokens[:, loss_offset + 1:org_len]) * mask[:, loss_offset+1:org_len]).sum() /
            mask[:, loss_offset + 1:org_len].sum())
        self.log_dict({
            "tr_loss": loss.item(),
            "accu": accu.item()
        },
                      prog_bar=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            if 'ln_' in name or 'bias' in name:
                print("Skip WD on {}".format(name))
                params.append({"params": [p], "weight_decay": 0.0})
            else:
                params.append({"params": [p]})
        optimizer = self.hparams.optimizer_cls(params)
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths)
        batch_size = lengths.shape[0]
        if batch_size != 1:
            enc_masks = torch.zeros(
                batch_size, max_len, dtype=torch.float).to(lengths.device)
            for e_id, src_len in enumerate(lengths):
                enc_masks[e_id, :src_len] = 1
        # for onnx conversion. ORT does not support pytorch tensor slice
        else:
            enc_masks = torch.ones(
                batch_size, max_len, dtype=torch.float).to(lengths.device)

        return enc_masks.bool()


    # @torch.no_grad()
    # def prepare_feature(self, wavs):
    #     device = wavs.device

    #     tokens = self.get_bshall_hubert_token(wavs)
    #     b, t = tokens.size()

    #     # input_tokens = torch.cat([mulan_tokens, eos_ids, w2v_tokens], dim=1)
    #     input_tokens = tokens

    #     return input_tokens


    @torch.no_grad()
    def prepare_feature(self, wavs, wav_lens):
        device = wavs.device
        tokens = self.get_hubert_token(wavs, wav_lens)

        return tokens


    @torch.no_grad()
    def get_quant(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2)
        return output

    # @torch.no_grad()
    # def get_bshall_hubert_token(self, x):
    #     return bshall_hubert_tokenization(
    #         frontend=None,
    #         bshall_hubert_model=self.requires["semantic"],
    #         wavs=x,
    #         centers=self.requires["centroids"],
    #         device=x.device,
    #     )

    @torch.no_grad()
    def get_hubert_token(self, wavs, wav_lens):
        return hubert_tokenization(
            hubert_model=self.requires["semantic"],
            wavs=wavs,
            wav_lens=wav_lens,
            centers=self.requires["centroids"],
            device=wavs.device,
        )