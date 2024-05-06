import logging
import os
import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.musiclm.utils.dist import local_zero_first
from recipes.music_dit.model.loss import MaskedMAELoss, MaskedSSIMLoss, MaskedMSELoss, MaskedMAELossDim2, sequence_mask
from recipes.music_dit.utils.infer_utils import save_wav
from recipes.umm.requires.model_initializer import init_stage3_dual_voc
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

LOSS_DICT = {
    "l1": MaskedMAELoss,
    "l1dim2": MaskedMAELossDim2,
    "l2": MaskedMSELoss,
    "ssim": MaskedSSIMLoss
}


def fix_flashattn_version(model_cls):
    hp = model_cls.keywords["hp"]
    if hp.use_window_mask and hp.flashattn_version != "2.3":
        print("WARN: try to change flashattn_version from '2' to '2.3' when use_window_mask=True")
        hp.flashattn_version = "2.3"
        model_cls.keywords["hp"] = hp
    return model_cls


class VoiceBoxModule(pl.LightningModule):

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        criterions,
        checkpointing=True,
        resume_ckpt_path = None,
        val_output_samples_dir="",
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.model = fix_flashattn_version(model_cls)()
        self.model = model_cls()
        
        self.criterion_dict = {}
        for criterion in criterions:
            self.criterion_dict[criterion] = LOSS_DICT[criterion]()

        self.requires = {}
        if checkpointing:
            self.model.gradient_checkpointing_enable()

        if resume_ckpt_path is not None:
            self.load_from_pretrained(resume_ckpt_path)

    def setup(self, stage: str) -> None:
        self.load_required_modules()
        
    def load_required_modules(self, ignore=()):
        for name, item in self.hparams.required_modules.items():
            if name in ignore: continue
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
                self.requires.update(initializer(hpath, local_rank=self.local_rank))
            elif isinstance(item, dict):
                initializer = item['initializer']
                if 'hpath' in item:
                    hpath = item['hpath']
                    self.requires.update(initializer(hpath, local_rank=self.local_rank))
                else:
                    self.requires.update(initializer(local_rank=self.local_rank))

    def load_from_pretrained(self, pretrained_path=None):
        rank_zero_info(f'Loading pre-trained model from checkpoint {pretrained_path}')
        with local_zero_first():
            local_path = download_checkpoint(pretrained_path, '.')

            ckpt_state_dict = torch.load(
                local_path, map_location=torch.device("cpu")
            )['state_dict']
            model_state_dict = self.model.state_dict()
            new_state_dict = {}

            for k in ckpt_state_dict:
                new_k = k.replace("model.", "") # saved model has prefix "model."
                if new_k in model_state_dict:
                    if ckpt_state_dict[k].shape != model_state_dict[new_k].shape:
                        rank_zero_info(f"Skip loading parameter: {k}, "
                                    f"required shape: {model_state_dict[new_k].shape}, "
                                    f"loaded shape: {ckpt_state_dict[k].shape}")
                    else:
                        new_state_dict[new_k] = ckpt_state_dict[k]
                else:
                    rank_zero_info(f"Dropping parameter {k}")

            self.model.load_state_dict(new_state_dict, strict=False)

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def prepare_input(self, batch, strip_txt_padding=False, Tctx=None):
        if 'wav' not in batch:
            batch['wav'] = batch['target_audio']
        if 'melae' in self.requires:
            batch['bn'] = self.requires['melae'].wav2token(batch['wav'])
        else:  # music vae
            wvae = self.requires['vocoder']
            h_ = wvae.encode(batch['wav'][:, None])
            batch['bn'], _, _ = wvae.sample(h_, deterministic=False)
            batch['bn'] = batch['bn'].transpose(1, 2)
            batch['seqlen'] = batch['seqlen'] * 10  # 12.5hz to 125hz

        if 'leadsheet_token' in batch:  # leadsheet2song
            batch['text_tokens'] = batch['leadsheet_token']
        elif 'leadsheet_tokens' in batch:  # leadsheet2song
            batch['text_tokens'] = batch['leadsheet_tokens']
        else:  # lyric2song or tts
            batch['text_tokens'] = batch['lyrics_tokens']
        batch['text_lens'] = (batch['text_tokens'] > 0).sum(-1)
        if strip_txt_padding:
            assert batch['text_tokens'].shape[0] == 1
            batch['text_tokens'] = batch['text_tokens'][:, batch['text_tokens'][0] > 0]
        if self.training:
            mask_token_id = 1
            masked_text_b = torch.rand_like(batch['text_tokens'][:, 0].float())
            masked_text_b = (masked_text_b[:, None] < 0.15).long()
            batch['text_tokens'] = batch['text_tokens'] * (1 - masked_text_b) + mask_token_id * masked_text_b

        Tmax = batch['bn'].shape[1]
        batch['seqlen'] = batch['seqlen'].clamp_max(Tmax)

        batch['bn_ctx_mask'] = torch.ones_like(batch['bn'][..., :1])
        batch['bn_ctx'] = batch['bn'].clone()
        if Tctx is None:
            Tctx = random.randint(0, Tmax // 2 - 1)
        batch['bn_ctx_mask'][:, Tctx:] = 0
        batch['bn_ctx'][:, Tctx:] = 0

        bsz, seqlen = batch['bn'].shape[0], batch['seqlen'].sum()
        loss_mask = sequence_mask(batch['seqlen'], Tmax, device=batch['seqlen'].device)
        return bsz, seqlen, loss_mask

    def prepare_predict_input(self, batch, strip_txt_padding=False, Tctx=None):
        if 'wav' not in batch:
            batch['wav'] = batch['target_audio']
        if 'melae' in self.requires:
            batch['bn'] = self.requires['melae'].wav2token(batch['wav'])
        else:  # music vae
            wvae = self.requires['vocoder']
            h_ = wvae.encode(batch['wav'][:, None])
            batch['bn'], _, _ = wvae.sample(h_, deterministic=False)
            batch['bn'] = batch['bn'].transpose(1, 2)
            batch['seqlen'] = batch['seqlen'] * 10  # 12.5hz to 125hz

        if 'leadsheet_token' in batch:  # leadsheet2song
            batch['text_tokens'] = batch['leadsheet_token']
        elif 'leadsheet_tokens' in batch:  # leadsheet2song
            batch['text_tokens'] = batch['leadsheet_tokens']
        else:  # lyric2song or tts
            batch['text_tokens'] = batch['lyrics_tokens']
        batch['text_lens'] = (batch['text_tokens'] > 0).sum(-1)
        if strip_txt_padding:
            assert batch['text_tokens'].shape[0] == 1
            batch['text_tokens'] = batch['text_tokens'][:, batch['text_tokens'][0] > 0]
        if self.training:
            mask_token_id = 1
            masked_text_b = torch.rand_like(batch['text_tokens'][:, 0].float())
            masked_text_b = (masked_text_b[:, None] < 0.15).long()
            batch['text_tokens'] = batch['text_tokens'] * (1 - masked_text_b) + mask_token_id * masked_text_b

        Tmax = batch['bn'].shape[1]
        batch['seqlen'] = batch['seqlen'].clamp_max(Tmax)

        batch['bn_ctx_mask'] = torch.ones_like(batch['bn'][..., :1])
        batch['bn_ctx'] = batch['bn'].clone()
        if Tctx is None:
            Tctx = random.randint(0, Tmax // 2 - 1)
        batch['bn_ctx_mask'][:, Tctx:] = 0
        batch['bn_ctx'][:, Tctx:] = 0

        bsz, seqlen = batch['bn'].shape[0], batch['seqlen'].sum()
        loss_mask = sequence_mask(batch['seqlen'], Tmax, device=batch['seqlen'].device)
        return bsz, seqlen, loss_mask

    def training_step(self, batch, batch_idx):
        bsz, seqlen, loss_mask = self.prepare_input(batch)

        pred, target = self.model(batch)

        loss_dict = {}
        loss = 0
        for loss_type, loss_func in self.criterion_dict.items():
            tmp_loss = loss_func(pred, target, loss_mask)
            loss_dict[loss_type] = tmp_loss.item()
            loss += tmp_loss

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {
                "loss": loss.item(),
                "bsz": bsz,
                "seqlen": seqlen
            }
            log_dict.update(loss_dict)
            log_dict["training/loss"] = log_dict["loss"]
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    @torch.no_grad()
    def inference(self, inputs, step):
        #with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        x = self.model.inference(inputs, step)
        return x

    predict = inference

    def plot_mel(self, mel, name):
        fig = plt.figure(figsize=(12, 8))
        plt.pcolor(mel.T)
        plt.savefig(f'{self.hparams.val_output_samples_dir}/{name}.png')
        plt.close(fig)

    def save_mel_wav(self, latents, name):
        if 'melae' in self.requires:
            mel = self.requires['melae'].latent2mel(latents)
            self.plot_mel(mel[0].float().cpu().numpy(), name=f'{name}')
            wav = self.melvoc(mel.transpose(1, 2), None).reshape(-1)
        else:
            wvae = self.requires['vocoder']
            wav = wvae.decode(latents.float().transpose(1, 2)).reshape(-1)
        save_wav(wav.float().cpu().numpy(),
                 f'{self.hparams.val_output_samples_dir}/{name}.wav')
        print(f"| saved wave to {self.hparams.val_output_samples_dir}/{name}")

    def validation_step(self, inputs, step):
        bsz, seqlen, loss_mask = self.prepare_input(
            inputs, strip_txt_padding=True, Tctx=torch.ones_like(inputs["seqlen"][0]))
        # inputs, strip_txt_padding=True, Tctx=inputs["seqlen"][0] // 3)
        diffusion_nfe = 20
        diffusion_sampler = "ddim"
        text_cfg_w = 4
        batch_num = inputs["bn"].shape[0]
        assert batch_num == 1

        inputs['bn_ctx'] = inputs['bn_ctx'].repeat(2, 1, 1)
        inputs['bn_ctx_mask'] = inputs['bn_ctx_mask'].repeat(2, 1, 1)
        inputs['text_lens'] = inputs['text_lens'].repeat(2)
        if 'style_text' in inputs:
            inputs['style_text'] = inputs['style_text'] + inputs['style_text']
        inputs['seqlen'] = inputs['seqlen'].repeat(2)
        inputs['text_tokens'] = torch.cat([
            inputs['text_tokens'], torch.ones_like(inputs['text_tokens'])], 0)

        lat = self.model.inference(
            inputs=inputs,
            timesteps=diffusion_nfe,
            sampler=diffusion_sampler,
            text_cfg_w=text_cfg_w)
        self.save_mel_wav(lat.transpose(1, 2), f'{self.trainer.global_step}/val{step:04d}_out')
        recording_wav_path = f'{self.hparams.val_output_samples_dir}/val{step:04d}_recording.wav'
        if not os.path.exists(recording_wav_path):
            self.save_mel_wav(inputs['bn'], f'val{step:04d}_gt')
            save_wav(inputs['wav'].float().cpu().numpy().reshape(-1), recording_wav_path)

    def prepare_output_dir(self):
        os.makedirs(f'{self.hparams.val_output_samples_dir}/{self.trainer.global_step}', exist_ok=True)
        with local_zero_first():
            local_path = download_checkpoint('hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/voc/v1/checkpoints/vocoder_last.ckpt', '.module_cache')
            self.melvoc = init_stage3_dual_voc(
                str(local_path),
                cache_dir='.module_cache', local_rank=self.local_rank)['mel_vocoder']

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.prepare_output_dir()

    def on_predict_epoch_start(self) -> None:
        super().on_predict_epoch_start()
        self.prepare_output_dir()

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        self.validation_step(batch, step=batch_idx)
