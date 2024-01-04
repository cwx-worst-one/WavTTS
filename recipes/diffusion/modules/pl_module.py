import os
import math
import time
import torch
import torchaudio
import soundfile as sf
import pytorch_lightning as pl
from einops import repeat

from pytorch_lightning.profilers import PassThroughProfiler
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from recipes.diffusion.models.diffusion_sampler import Sampler
from recipes.soundstream.utils.sample_pool import SamplePool, OfflineFeatureSamplePool
from recipes.diffusion.utils.utils import random_side_mask

from recipes.musiclm.utils.dist import local_zero_first
from recipes.diffusion.utils.utils import download_checkpoint

from samantha.utils.custom_flops_profilers.diffusion_flops_profiler import TNTDiffusionNetworkFLOPsCounter
from samantha.utils.flops_profiler import FlopsProfiler
from samantha.utils.model_metric import ModelMetric


torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DiffusionModule(pl.LightningModule):
    def __init__(
            self, 
            seed,
            diffusion_model,
            lora,
            target_dim,
            num_chunks,
            duration,
            noise_range,
            tokenizer_sample_rate,
            diffusion_sample_rate,
            sample_pool_size,
            train_batch_size,
            val_batch_size,
            vc,
            optimizer_cls, 
            scheduler_cls,
            pretrained_path,
            required_modules,
            val_output_samples_dir,
            flops_fn,
            online_feature,
        ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=['diffusion_model'])
   
        self.loss_function = torch.nn.MSELoss()
        
        self.requires = {}
        self.model = diffusion_model 
        # load pretrained model
        if self.hparams.pretrained_path is not None:
            self.load_from_pretrained(self.hparams.pretrained_path)
        # lora
        if self.hparams.lora:
            for k, v in self.model.named_parameters():
                if 'lora' not in k and 'vc_null_embedding' not in k and 'vc_context_embed' not in k and 'sep_embedding' not in k:
                    v.requires_grad = False
                else:
                    rank_zero_info(f'Enable gradient for {k}')
   
        self.sampler = Sampler(
            in_channels=target_dim,
            window_length=duration,
            vocoder_hz=125 if diffusion_sample_rate==24000 else 147
        )
        self.current_step = 0
        if sample_pool_size > 0:
            if online_feature:
                self.sample_pool = SamplePool(
                    data_samplerate=diffusion_sample_rate,
                    cache_size=sample_pool_size,
                    batch_size=train_batch_size,
                    length_samples=int(duration*diffusion_sample_rate),
                    silence_prob=0.05,
                )
            else:
                self.sample_pool = OfflineFeatureSamplePool(
                    cache_size=sample_pool_size,
                    batch_size=train_batch_size,
                    silence_prob=0,
                )
        if diffusion_sample_rate != tokenizer_sample_rate:
            self.resampler = torchaudio.transforms.Resample(diffusion_sample_rate, tokenizer_sample_rate)
    
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

    def load_required_modules(self, ignore=()):
        for name, item in self.hparams.required_modules.items():
            if name in ignore: continue
            if isinstance(item, (list, tuple)):
                hpath, initializer = item
            elif isinstance(item, dict):
                hpath = item['hpath']
                initializer = item['initializer']
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def setup(self, stage: str) -> None:
        self.load_required_modules()
        # set device for sampler
        self.sampler.set_device(torch.device(f"cuda:{self.local_rank}"))

        # set torch seed for randomness
        torch.manual_seed(self.hparams.seed + self.global_rank)
        if getattr(self, "resampler", False):
            self.resampler = self.resampler.to(torch.device(f"cuda:{self.local_rank}"))

        flops_fn = getattr(self.hparams,"flops_fn","deepspeed")
        if flops_fn =="deepspeed":
            flops_fn = FlopsProfiler(self.model)
        elif flops_fn == "custom":
            flops_fn = TNTDiffusionNetworkFLOPsCounter(
                depth = self.model.depth,
                input_dim = self.model.input_dim,
                feature_dim = self.model.feature_dim,
                context_dim = self.model.context_dim,
                segment_size = self.model.segment_size,
                segment_stride= self.model.segment_stride,
                training=self.trainer.state.stage
            )
        else:
            raise NotImplementedError(f"no support flops_fn '{flops_fn}'")
        self.model.flops_fn = flops_fn
        self.model_metric = ModelMetric(self.trainer.precision,self.model)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

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
    def get_vc_condition_emb(self, x):
        emb = self.get_vocoder_embs(x)
        return emb.detach()
    @torch.no_grad()
    def get_condition_tokens(self, x):
        self.requires['Stage3'].eval()
        vq_ids = self.requires['Stage3'].wav2token(x)
        return vq_ids.detach()

    @torch.no_grad()
    def get_condition_embs(self, x):
        self.requires['Stage3'].eval()
        encoded_feature = self.requires['Stage3'].model.wav2embed(x)
        shared_encoder_output = self.requires['Stage3'].model.shared_encoder.forward_to_vq(
            encoded_feature, vq=self.requires['Stage3'].model.vq
        )
        # output keys: ['vq_states', 'vq_ids', 'vq_loss', 'vq_emb']
        vq_emb = shared_encoder_output['vq_emb']
        return vq_emb.detach()

    @torch.no_grad()
    def get_vocoder_embs(self, x):
        # input x has shape (b, c, t)
        self.requires['vocoder'].eval()
        encoder_out = self.requires['vocoder'].encode(x)
        sample, _, _ = self.requires['vocoder'].sample(encoder_out,  deterministic=False) # TODO: check wethear deterministic should be True or False
        return sample.detach()

    @torch.no_grad()
    def vocoder_embs_to_wav(self, x):
        # input x has shape (b, c, t)
        self.requires['vocoder'].eval()
        wav = self.requires['vocoder'].decode(x)
        return wav.detach()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[litmodule]TNTDiffusionNetwork.pre_feature"):
            torch.cuda.synchronize()
            start_ts = time.perf_counter()
            # temp treatment for mcc
            if self.hparams.online_feature:
                if type(batch) is list:
                    batch = {'audio': batch[0]}
                elif 'target_audio' in batch:
                    batch['audio'] = batch['target_audio'] 
                if len(batch['audio'].shape) < 3:
                    batch['audio'] = batch['audio'][:, None, :] # bs, seq_len -> bs, ch1, seq_len
                if getattr(self, "sample_pool", False):
                    batch = {'audio': self.sample_pool.process(batch['audio'])}
                if getattr(self, "resampler", False):
                    condition_audio = self.resampler(batch['audio'].float())
                    condition_audio = condition_audio.mean(dim=1, keepdim=True)
                else:
                    condition_audio = batch['audio']
                
                with torch.autocast(device_type="cuda", enabled=False):
                    # context
                    condition_tokens = self.get_condition_tokens(condition_audio)
                    # target
                    vocoder_embs = self.get_vocoder_embs(batch['audio'].float())
                    if self.hparams.vc:
                        vc_condition_emb = self.get_vc_condition_emb(batch['vocal_audio'].float())
                        vc_condition_tokens = self.get_condition_tokens(batch['vocal_audio'].float())
                        
                        # concat vc condition at the front
                        condition_tokens = torch.cat([vc_condition_tokens, condition_tokens], dim=-1)
                        vocoder_embs = torch.cat([vc_condition_emb, vocoder_embs], dim=-1)
            else:
                batch = self.sample_pool.process(batch)
                # context
                condition_tokens = batch["condition_tokens"]
                # target
                vocoder_embs = batch["vocoder_embs"]
                
                if self.hparams.vc:
                    vc_condition_emb = batch["vc_condition_emb"]
                    vc_condition_tokens = batch["vc_condition_tokens"]
                    # concat vc condition at the front
                    condition_tokens = torch.cat([vc_condition_tokens, condition_tokens], dim=-1)
                    vocoder_embs = torch.cat([vc_condition_emb, vocoder_embs], dim=-1)
                
            # get training target
            b, d, l = vocoder_embs.shape
            et = torch.randn_like(vocoder_embs)
        
            # define noise level 
            lower_bound = self.hparams.noise_range[0]
            upper_bound = self.hparams.noise_range[1]
            t = torch.rand(size=[b, 1, self.hparams.num_chunks], device=self.device, dtype=vocoder_embs.dtype)
            t = (lower_bound - upper_bound) * t + upper_bound
            # extent t to l, use clone to avoid call by reference
            t = repeat(t, 'b 1 n -> b 1 (n l)', l=l).clone().detach()

            if self.hparams.vc:
                t = torch.cat([torch.zeros_like(t)[..., :vc_condition_emb.shape[-1]], t[..., vc_condition_emb.shape[-1]:]], dim=-1)
            else:
                # multi task
                # TODO: merge vc into this part
                task_dice = torch.randint(0, 2, (1,)).item()
                is_causal = False
                # generation
                if task_dice == 0:
                    # causal or non-causal
                    if torch.rand(1).item() < 0:
                        is_causal = True
                # inpainting / outpainting
                elif task_dice == 1:
                    t = random_side_mask(t, min_mask_percentage=0.2, max_mask_percentage=0.8)

                # WIP causal continuation, need to implement mask
                elif task_dice == 2:
                    pass

            angles = math.pi /2. * t
            alphas, deltas = torch.cos(angles), torch.sin(angles)
        
            xt = alphas * vocoder_embs + deltas * et
            vt = alphas * et - deltas * vocoder_embs
            torch.cuda.synchronize()
            end_ts = time.perf_counter()
        # model_metric
        num_tokens = b * l

        self.model_metric.num_tokens += num_tokens
        if self.trainer.global_step % self.trainer.log_every_n_steps==0:
            self.model_metric.update(0,self.trainer.state.stage)

        with self.profiler.profile("[litmodule]TNTDiffusionNetwork.forward"):
            vt_pred = self.model(
                xt, 
                t, 
                semantic_context=condition_tokens.detach(),
                vc_context=vc_condition_emb.mean(dim=-1, keepdim=True).permute(0, 2, 1).detach() if self.hparams.vc else None,
                is_causal=is_causal,
            )

            with torch.autocast(device_type="cuda", enabled=False):
                # unweighted_loss = self.loss_function(vt_pred[:, :, vc_condition_emb.shape[-1]:].float(), vt[:, :, vc_condition_emb.shape[-1]:].float())
                unweighted_loss = self.loss_function(vt_pred.float(), vt.float())
                unweighted_loss = torch.mean(unweighted_loss)
                loss = torch.mean(unweighted_loss)
        if isinstance(self.model.flops_fn,TNTDiffusionNetworkFLOPsCounter):
            self.model.flops_fn(b,l, condition_tokens.shape[1], vc_condition_emb.shape[1] if self.hparams.vc else 0)
            
        flops = self.model.flops_fn.get_total_flops()
        
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            log_dict = {
                "training/loss": loss, 
                "training/unweighted_loss": unweighted_loss,
            }
            log_dict["training/bs"] = b
            log_dict["training/seqlen"] = l
            log_dict["training/prefeautre"] = round(end_ts - start_ts, 3)
            log_dict.update(self.model_metric.compute(self.trainer.global_step))
            self.log_dict(log_dict,prog_bar=True,sync_dist=True)
        self.current_step += 1

        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        # create folder based on current epoch
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.current_step}", exist_ok=True
        )
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}", exist_ok=True
        )
        
        self.val_output_dict = {}
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.autocast(device_type="cuda", enabled=False):
            if self.hparams.online_feature:
                if type(batch) is list:
                    batch = {'audio': batch[0]}
                elif 'target_audio' in batch:
                    batch['audio'] = batch['target_audio']
                if len(batch['audio'].shape) < 3:
                    batch['audio'] = batch['audio'][:, None, :] # bs, seq_len -> bs, ch1, seq_len
                
                # context
                batch['audio'] = batch['audio'][:self.hparams.val_batch_size, :int(self.hparams.duration*self.hparams.diffusion_sample_rate)]

                # pad to int(self.hparams.duration*self.hparams.diffusion_sample_rate)
                if batch['audio'].shape[-1] < int(self.hparams.duration*self.hparams.diffusion_sample_rate):
                    pad_len = int(self.hparams.duration*self.hparams.diffusion_sample_rate) - batch['audio'].shape[-1]
                    batch['audio'] = torch.nn.functional.pad(batch['audio'], (0, pad_len), 'constant', 0)

                if getattr(self, "resampler", False):
                    condition_audio = self.resampler(batch['audio'] .float())
                    condition_audio = condition_audio.mean(dim=1, keepdim=True)
                else:
                    condition_audio = batch['audio']

                condition_tokens = self.get_condition_tokens(condition_audio)

                if self.hparams.vc:
                    batch['vocal_audio'] = batch['vocal_audio'][:self.hparams.val_batch_size]
                    vc_condition_emb = self.get_vc_condition_emb(batch['vocal_audio'].float())
                    vc_condition_tokens = self.get_condition_tokens(batch['vocal_audio'].float())
                    condition_tokens = torch.cat([vc_condition_tokens, condition_tokens], dim=-1)
            else:
                # NOTE: get list(tensor), need 
                # FIXME: item may have different batch_size in 'condition_tokens' and 'vocoder_embs'
                condition_tokens = batch["condition_tokens"][:self.hparams.val_batch_size, 0]
                gt_vocoder_embs = batch["vocoder_embs"][:self.hparams.val_batch_size, 0]
                assert len(condition_tokens)==len(gt_vocoder_embs), (condition_tokens.shape,gt_vocoder_embs.shape)
                # TODO
                if self.hparams.vc:
                    vc_condition_emb = batch["vc_condition_emb"][:self.hparams.val_batch_size]
                    vc_condition_tokens = batch["vc_condition_tokens"][:self.hparams.val_batch_size]
                    condition_tokens = torch.cat([vc_condition_tokens, condition_tokens], dim=-1)
            
            # diffusion sampling
            pred_emb = self.sampler(
                model=self.model, 
                semantic_context=condition_tokens,
                num_items=condition_tokens.shape[0], # batch size: how many samples to generate
                num_chunks=self.hparams.num_chunks,
                num_steps=25, # diffusion steps
                bf16_portion=0.0,
                angle_schedule='linear',
                schdeule_slope=2.5,
                classifier_free_guidance=2.5,
                vc_context=vc_condition_emb.mean(dim=-1, keepdim=True).permute(0, 2, 1) if self.hparams.vc else None,
                vc_prefix=vc_condition_emb if self.hparams.vc else None,
            )
            # generate audio
            wavs_g = self.vocoder_embs_to_wav(pred_emb.float())

        # save the output wavs
        if self.hparams.online_feature:
            for idx, (gt, wav) in enumerate(zip(batch['audio'], wavs_g)):
                num_files = len(os.listdir(f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}")) // 2
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}/{num_files}.wav",
                    wav.cpu().numpy().T,
                    self.hparams.diffusion_sample_rate,
                )
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}/{num_files}_gt.wav",
                    gt.cpu().numpy().T,
                    self.hparams.diffusion_sample_rate,
                )
        else:
            gt_wavs = self.vocoder_embs_to_wav(gt_vocoder_embs.float())
            for idx, (gt_wav, wav) in enumerate(zip(gt_wavs, wavs_g)):
                num_files = len(os.listdir(f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}")) // 2
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}/{num_files}.wav",
                    wav.cpu().numpy().T,
                    self.hparams.diffusion_sample_rate,
                )
                sf.write(
                    f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}/{num_files}_gt.wav",
                    gt_wav.cpu().numpy().T,
                    self.hparams.diffusion_sample_rate,
                )
        # TODO: FAD?




