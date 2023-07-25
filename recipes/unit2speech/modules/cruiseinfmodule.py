import cruise
import torch
import numpy as np
import functools
import os
import random
import time
import soundfile as sf

from recipes.unit2speech.models.diffusion.dprtnet import DPRTNet
from recipes.unit2speech.models.diffusion.v_diffusion import ARVSampler, Diffusion
from recipes.unit2speech.models.ema import EMAModel
from recipes.unit2speech.modules.utils import get_config_from_file, remove_ddp_module
from recipes.unit2speech.modules.datamodule import U2SDataModule
from recipes.unit2speech.models.diffusion.embedder import FrozenSSLEmbedder


class U2SInfModule(cruise.CruiseModule):

    def __init__(
        self,
        # Model params
        input_dim=256,
        feature_dim=1024,
        num_blocks=8,
        segment_size=64,
        segment_stride=32,
        context_dim=512,
        dropout=0.,
        intra_seq2seq='lstm',
        inter_seq2seq='lstm',
        predict_xstart=False,
        batch_size=1,
        diffusion_steps=0,
        num_chunks=20,
        chunk_length=125, # 20*125=2500 for training
        sampling_length=2500,
        autoencoder="autoencoder",
        autoencoder_path='',
        autoencoder_config='',
        model_ckpt_dir='./tmp',
        use_ema=True,
        classifier_free_guidance=1,
        test_prompt=False,
        cond_embedder='ssl',
    ):
        # create model and diffusion
        super().__init__()
        self.save_hparams()

    def setup(self, stage) -> None:

        self.output_dir = os.path.join(self.hparams.model_ckpt_dir, 'samples')
        os.makedirs(self.output_dir, exist_ok=True)

        # conditional on audio/text embedding
        embedder = FrozenSSLEmbedder(device=self.trainer.root_device)
        embedder = embedder.to(self.trainer.root_device)
        self.rank_zero_print("Condition Embedder ({}) Params: {:.4f}M".format(self.hparams.cond_embedder, sum(p.numel() for p in embedder.parameters()) / 1e6))

        self.embedder = embedder

        # prepare diffusion model
        self.model = DPRTNet(
            input_dim=self.hparams.input_dim,
            feature_dim=self.hparams.feature_dim,
            num_blocks=self.hparams.num_blocks,
            num_chunks=self.hparams.num_chunks,
            segment_size=self.hparams.segment_size,
            segment_stride=self.hparams.segment_stride,
            diffusion_steps=self.hparams.diffusion_steps,
            dropout=self.hparams.dropout,
            intra_seq2seq=self.hparams.intra_seq2seq,
            inter_seq2seq=self.hparams.inter_seq2seq,
        )
        if self.hparams.diffusion_steps <= 0:
            self.diffusion = Diffusion(
                steps=self.hparams.diffusion_steps,
                num_chunks=self.hparams.num_chunks,
                chunk_length=self.hparams.chunk_length,
                training=True,
            )
        else:
            self.diffusion = ARVSampler(
                self.model,
                in_channels=self.hparams.input_dim,
                length=self.hparams.num_chunks*self.hparams.chunk_length,
                num_splits=self.hparams.num_chunks,
            )

        self.rank_zero_print("Diffusion Model Params: {:.4f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6))

        # prepare audio encoder
        if self.hparams.autoencoder_config:
            hp = get_config_from_file(f"{self.hparams.autoencoder_config}").hparams
        else:
            hp = get_config_from_file(f"{self.hparams.autoencoder}/config.yaml").hparams
        if self.hparams.autoencoder == 'autoencoder':
            from recipes.unit2speech.models.autoencoder.autoencoder_kl import AutoencoderKL
            autoencoder = AutoencoderKL(hp, stage='enc')
            ckpt = torch.load(self.hparams.autoencoder_path, map_location='cpu')
            state = remove_ddp_module(ckpt['G'])
            for k in state:
                if 'encoder.cnt' in k:
                    del state[k]
                    break
            autoencoder.load_state_dict(state)
            autoencoder.eval()
        else:
            raise NotImplementedError
        if autoencoder is not None:
            self.rank_zero_print("Audio Encoder ({}) Params: {:.4f}M".format(self.hparams.autoencoder, sum(p.numel() for p in autoencoder.parameters()) / 1e6))

        self.autoencoder = autoencoder

        # load checkpoint 

        if self.hparams.use_ema:
            ema_ckpt = torch.load(os.path.join(self.hparams.model_ckpt_dir, 'ema_diffusion_model.pt'), map_location='cpu')
            ema_state_dict = remove_ddp_module(ema_ckpt)
        ckpt = torch.load(os.path.join(self.hparams.model_ckpt_dir, 'diffusion_model.pt'), map_location='cpu')
        state_dict = remove_ddp_module(ckpt)
        init_dict = self.model.state_dict()
        for k in init_dict:
            if self.hparams.use_ema and k in ema_state_dict:
                state_dict[k] = ema_state_dict[k]
            if k not in state_dict:
                state_dict[k] = init_dict[k]
        for k in state_dict:
            if k not in init_dict:
                self.rank_zero_print(k, 'in checkpoint not in model!', flush=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.rank_zero_print("Diffusion Model Params: {:.4f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))

    def training_step(self, batch, batch_idx):
        # dummy
        return None

    def on_predict_start(self) -> None:
        self.classifier_free_guidance = self.hparams.classifier_free_guidance
        self.i = 0
        self.cnt_audios = 0
        random.seed(0)
        self.embedder = self.embedder.to(self.trainer.root_device) # ADDED
        self.rank_zero_print("start sampling...")

    def on_predict_end(self) -> None:
        self.rank_zero_print("sampling complete")
    
    def predict_step(self, batch, batch_idx):
        (wavs, wav_lens), tokens, paths = batch
        if tokens is None:
            tokens = self.embedder((wavs, wav_lens))
            token_lens = torch.tensor([tokens.shape[-1],]).long()
        else:
            tokens, token_lens = tokens
        wavs, tokens, token_lens = wavs.to(self.trainer.root_device), tokens.to(self.trainer.root_device), token_lens.to(self.trainer.root_device)
        if self.hparams.test_prompt and self.i % 2 == 0:
            prompt = wavs
            self.i += 1
            return
        self.i += 1
        zs, z_lens = self.autoencoder(wavs, wav_lens)
        if not self.hparams.test_prompt:
            prompt = wavs
        start_time = time.time()
        sample = self.diffusion(
            self.hparams.batch_size,
            self.hparams.diffusion_steps,
            xT=None,
            skip_steps=0,
            show_progress=True,
            classifier_free_guidance=self.hparams.classifier_free_guidance,
            model_kwargs={"context": (tokens, token_lens),
                        "z_lens": z_lens,
                        "prompt": prompt[..., :3 * 24000]}
        )
        dpd_time = time.time() - start_time
        #accelerator.print('RTF: ', dpd_time / (wav_lens[0] / 24000), flush=True)
        start_time = time.time()
        for i in range(len(sample)):
            wav_len = wav_lens[i]
            wav_name = paths[i].split('/')[-1].split('.')[0]
            audio = self.autoencoder.decode(sample[i, :, :z_lens[i]][None]).detach().cpu()
            audio = audio / audio.abs().max(-1, keepdim=True)[0] * 0.6
            audio = audio.numpy().ravel()
            filename = f'{self.output_dir}/{wav_name}.wav'
            sf.write(filename, audio, 24000)
            ori_sample = wavs[i].cpu()[..., :wav_len]
            ori_sample = ori_sample / ori_sample.abs().max(-1, keepdim=True)[0] * 0.6
            audio_ori = ori_sample.numpy().ravel()
            filename = f'{self.output_dir}/{wav_name}_gt.wav'
            sf.write(filename, audio_ori, 24000)
            if self.hparams.test_prompt:
                audio_prompt = prompt[i, :3 * 24000]
                audio_prompt = audio_prompt / audio_prompt.abs().max(-1, keepdim=True)[0] * 0.6
                filename = f'{self.output_dir}/{wav_name}_prompt.wav'
                sf.write(filename, audio_prompt.cpu().numpy().ravel(), 24000)
        self.cnt_audios += len(sample) * self.trainer.world_size
        torch.cuda.empty_cache()


if __name__ == '__main__':
    cli = cruise.CruiseCLI(U2SInfModule,
                    trainer_class=cruise.CruiseTrainer,
                    datamodule_class=U2SDataModule)
    cfg, trainer, model, datamodule = cli.parse_args()
    predictions = trainer.predict(model, predict_dataloader=datamodule.predict_dataloader())