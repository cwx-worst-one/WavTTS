import os
import math
import torch
import torchaudio
import soundfile as sf
import pytorch_lightning as pl
from collections import OrderedDict
from einops import rearrange, repeat

from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.diffusion.models.diffusion_mss import ARVSampler
from recipes.soundstream.utils.sample_pool import SamplePool

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import torch
import numpy as np
import pyloudnorm as pyln

class AugPool():
    def __init__(
        self, 
        sources=['vocals', 'drums', 'bass', 'other'],
        cache_size=50,
        batch_size=10,
        length_samples=44100,
    ):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.length_samples = length_samples
        self.data = {source: [] for source in sources}
        self.meter = pyln.Meter(44100)

    def _process_pool(self, batch):
        for key in batch:
            for audio in batch[key]:
                if key not in self.data:
                    continue
                # add
                if len(self.data[key]) == 0:
                    self.data[key].append(audio)
                else:
                    db = self.meter.integrated_loudness(audio.detach().cpu().numpy().T)
                    if db < -50 or np.isneginf(db): 
                        continue
                    self.data[key].append(audio)
                # remove
                if len(self.data[key]) > self.cache_size:
                    self.data[key].pop(0)

    def _batch_resample(self):
        new_batch = {source: [] for source in self.data.keys()}
        for i in range(self.batch_size):
            for source in self.data.keys():
                # dice 5 percent use silent audio
                if random.random() < 0.05:
                    new_batch[source].append(torch.torch.zeros_like(self.data[source][0][:, :self.length_samples]))
                else:
                    random_audio = random.choice(self.data[source])
                    start = random.randint(0, random_audio.shape[-1] - self.length_samples)
                    new_batch[source].append(random_audio[:, start:start+self.length_samples])
    
        for source in new_batch.keys():
            new_batch[source] = torch.stack(new_batch[source])
        return new_batch
    
    def process(self, batch):
        self._process_pool(batch)
        return self._batch_resample()

class PairProcessor():
    def __init__(
        self, 
        sources=['vocals', 'drums', 'bass', 'other'],
        target_source='vocals',
    ):
        self.sources = sources
        self.target_source = target_source

    def process(self, batch):
        new_batch = {'mixture': [], 'target': []}

        for key in batch:
            if key not in self.sources:
                continue
            if key == self.target_source:
                new_batch['target'] = batch[key]

            new_batch['mixture'].append(batch[key])

        new_batch['mixture'] = torch.sum(torch.stack(new_batch['mixture']), dim=0)
        return new_batch


class DiffusionModule(pl.LightningModule):
    def __init__(
            self, 
            seed,
            stage,
            diffusion_model,
            target_dim,
            num_chunks,
            chunk_length,
            noise_range,
            sample_rate,
            asset_dir,
            sources,
            target_source,
            sample_pool_size,
            train_batch_size,
            sample_length,
            optimizer_cls, 
            scheduler_cls,
            vocoder_model,
            val_output_samples_dir,
        ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=['diffusion_model'])
        self.stage = stage
        self.loss_function = torch.nn.MSELoss()
        
        self.model = diffusion_model 
        self.sampler = ARVSampler(
            in_channels=target_dim,
            length=num_chunks*chunk_length,
            num_splits=num_chunks,
        )
        # custom recorder for training step due to GAN training
        self.current_step = 0

        # sample pool for efficient training
        if stage == 'pretrain':
            self.sample_pool = SamplePool(
            cache_size=sample_pool_size,
            batch_size=train_batch_size,
            length_samples=int(sample_length),
            silence_prob=0.05,
        )
        elif stage == 'finetune':
            self.aug_pool = AugPool(
                sources,
                sample_pool_size,
                train_batch_size,
                int(sample_length),
            )
            self.pair_processor = PairProcessor(
                sources=sources,
                target_source=target_source,
            )

    def on_fit_start(self):
        # init required models
        self.vocoder_model = init_vocoder(
            checkpoint_path=self.hparams.vocoder_model['model_path'],
            local_rank=self.local_rank,
            cache_dir=self.hparams.asset_dir,
            sample_rate=self.hparams.sample_rate,
        )
        # set device for sampler
        self.sampler.set_device(self.device)

        # set torch seed for randomness
        torch.manual_seed(self.hparams.seed + self.global_rank)
    
    def _divide_params_group(self, model):
        no_decay = [
            "bn",
            "bias",
            "norm"
            "rotary",
            "embedding",
            ".g", # g in RMSNorm
        ]

        base_params = []
        no_decay_params = []
        for name, param in model.named_parameters(): 
            _found = False
            for k in no_decay:
                if k in name:
                    no_decay_params.append(param)
                    _found = True
                    break
            if not _found:
                base_params.append(param)

        return base_params, no_decay_params

    def configure_optimizers(self):
        base_params, no_decay_params = self._divide_params_group(self.model)

        optimizer = self.hparams.optimizer_cls(
            [{"params": base_params}, {"params": no_decay_params, "weight_decay": 0.0}],
        )
        scheduler = self.hparams.scheduler_cls(optimizer)
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_vocoder_embs(self, x):
        # input x has shape (b, c, t)
        self.vocoder_model["vocoder"].eval()
        encoder_out = self.vocoder_model["vocoder"].encode(x)
        sample, _, _ = self.vocoder_model["vocoder"].sample(encoder_out,  deterministic=False) # TODO: check wethear deterministic should be True or False
        return sample.detach()

    @torch.no_grad()
    def vocoder_embs_to_wav(self, x):
        # input x has shape (b, c, t)
        self.vocoder_model["vocoder"].eval()
        wav = self.vocoder_model["vocoder"].decode(x)
        return wav.detach()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        if self.stage == 'pretrain':
            batch = self.sample_pool.process(batch['audio'])

        elif self.stage == 'finetune':
            batch = self.aug_pool.process(batch) 
            batch = self.pair_processor.process(batch)
            

        with torch.autocast(device_type="cuda", enabled=False):
            # target
            if self.stage == 'pretrain':
                mixture_vocoder_embs = self.get_vocoder_embs(batch.float())
                target_vocoder_embs = mixture_vocoder_embs.clone()
            elif self.stage == 'finetune':
                mixture_vocoder_embs = self.get_vocoder_embs(batch['mixture'].float())
                target_vocoder_embs = self.get_vocoder_embs(batch['target'].float())

            # diffusion training
            b, d, l = target_vocoder_embs.shape
            et = torch.randn_like(target_vocoder_embs)
            # noise level ensemble
            lower_bound = self.hparams.noise_range[0]
            upper_bound = self.hparams.noise_range[1]
            t = torch.rand(size=[b, 1, self.hparams.num_chunks], device=self.device, dtype=target_vocoder_embs.dtype)
            t = (lower_bound - upper_bound) * t + upper_bound
        
            t = repeat(t, 'b 1 n -> b 1 (n l)', l=self.hparams.chunk_length)

            angles = math.pi /2. * t
            alphas, deltas = torch.cos(angles), torch.sin(angles)
        
            xt = alphas * target_vocoder_embs + deltas * et
            # concate with mixture emb
            xt = torch.cat([xt, mixture_vocoder_embs], dim=1)
            vt = alphas * et - deltas * target_vocoder_embs
        
        vt_pred = self.model(
            xt, 
            t, 
            context=mixture_vocoder_embs.permute(0, 2, 1).detach()
        )

        with torch.autocast(device_type="cuda", enabled=False):
            unweighted_loss = self.loss_function(vt_pred.float(), vt.float())
            loss = torch.mean(unweighted_loss)

        self.log_dict(
            {
                "training/loss": loss, 
                "train_loss": loss, 
                "unweighted_loss": torch.mean(unweighted_loss),
                "step": self.current_step
            },
            prog_bar=True,
            sync_dist=True,
        )
        self.current_step += 1

        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # create folder based on current epoch
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.current_step}", exist_ok=True
        )
        self.val_output_dict = {}
        return

    def validation_step(self, batch, batch_idx):
        with torch.autocast(device_type="cuda", enabled=False):
            if self.stage == 'pretrain':
                batch = batch['audio'][..., :int(self.hparams.sample_length)]
                mixture_vocoder_embs = self.get_vocoder_embs(batch.float())
            elif self.stage == 'finetune':
                mixture = []
                target = []
                for key in batch:
                    if key in self.hparams.sources:
                        stem = batch[key][:, :, :int(self.hparams.sample_length)]
                        mixture.append(stem)
                        if key == self.hparams.target_source:
                            target = stem
                mixture = torch.sum(torch.stack(mixture), dim=0)

                batch['mixture'] = mixture
                batch['target'] = target
                # context
                mixture_vocoder_embs = self.get_vocoder_embs(batch['mixture'].float())

            # diffusion sampling
            pred_emb = self.sampler(
                model=self.model, 
                context=mixture_vocoder_embs,
                num_items=mixture_vocoder_embs.shape[0], # batch size: how many samples to generate
                num_chunks=self.hparams.num_chunks,
                num_steps=20, # diffusion steps
                bf16_portion=0.0,
                start=None,
                show_progress=False,
                angle_schedule='linear',
                schdeule_slope=2.5,
                classifier_free_guidance=2.5,
            )
            # generate audio
            wavs_g = self.vocoder_embs_to_wav(pred_emb[:, :self.hparams.target_dim].float())

        # save the output wavs
        if self.stage == 'finetune' and self.local_rank == 0:
            for idx, (gt, mix, wav) in enumerate(zip(batch['target'], batch['mixture'], wavs_g)):
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{idx}.wav",
                    wav.cpu().numpy().T,
                    self.hparams.sample_rate,
                )
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{idx}_mix.wav",
                    mix.cpu().numpy().T,
                    self.hparams.sample_rate,
                )
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{idx}.gt.wav",
                    gt.cpu().numpy().T,
                    self.hparams.sample_rate,
                )

        # TODO: FAD?



