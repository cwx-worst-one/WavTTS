import os
import math
import torch
import torchaudio
import soundfile as sf
import pytorch_lightning as pl
from einops import repeat

from pytorch_lightning.utilities.rank_zero import rank_zero_info
from recipes.diffusion.models.diffusion_sampler import Sampler as Sampler
from recipes.soundstream.utils.sample_pool import SamplePool
from recipes.diffusion.utils.utils import random_side_mask

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DiffusionModule(pl.LightningModule):
    def __init__(
            self, 
            seed,
            diffusion_model,
            consistency_model,
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
        ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=['diffusion_model', 'consistency_model'])
   
        self.loss_function = torch.nn.MSELoss(reduction='none')
        
        self.requires = {}
        self.diffusion_model = diffusion_model 
        self.consistency_model = consistency_model

        # load pretrained model
        if self.hparams.pretrained_path is not None:
            self.load_from_pretrained(self.diffusion_model, self.hparams.pretrained_path)
            self.load_from_pretrained(self.consistency_model, self.hparams.pretrained_path)

        for k, v in self.diffusion_model.named_parameters():
            v.requires_grad = False

        # lora
        if self.hparams.lora:
            for k, v in self.diffusion_model.named_parameters():
                if 'lora' not in k and 'vc_null_embedding' not in k and 'vc_context_embed' not in k and 'sep_embedding' not in k:
                    v.requires_grad = False
                else:
                    rank_zero_info(f'Enable gradient for {k}')
   
        self.sampler = Sampler(
            in_channels=target_dim,
            duration=duration,
            num_splits=num_chunks,
            vocoder_hz=125 if diffusion_sample_rate==24000 else 147
        )
        self.current_step = 0
        if sample_pool_size > 0:
            self.sample_pool = SamplePool(
                data_samplerate=diffusion_sample_rate,
                cache_size=sample_pool_size,
                batch_size=train_batch_size,
                length_samples=int(duration*diffusion_sample_rate),
                silence_prob=0.05,
            )
        if diffusion_sample_rate != tokenizer_sample_rate:
            self.resampler = torchaudio.transforms.Resample(diffusion_sample_rate, tokenizer_sample_rate)
    
    def load_from_pretrained(self, model, pretrained_path=None):
        rank_zero_info(f'Loading pre-trained model from checkpoint {pretrained_path}')
        ckpt_state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )['state_dict']
        model_state_dict = model.state_dict()
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

        model.load_state_dict(new_state_dict, strict=False)

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
        base_params, no_decay_params = self._divide_params_group(self.consistency_model)

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

    def get_scalings_for_boundary_condition_discrete(self, t):
        self.sigma_data = 0.5       # Default: 0.5
        dominator = 0.00001
        
        # By dividing 0.1: This is almost a delta function at t=0.     
        c_skip = self.sigma_data**2 / (
                (t / dominator) ** 2 + self.sigma_data**2
            )
        c_out = (( t / dominator)  / ((t / dominator) **2 + self.sigma_data**2) ** 0.5)
        return c_skip, c_out

    def training_step(self, batch, batch_idx):        
        # temp treatment for mcc
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

        angles = math.pi /2. * t
        alphas, deltas = torch.cos(angles), torch.sin(angles)
    
        xt = alphas * vocoder_embs + deltas * et
        vt = alphas * et - deltas * vocoder_embs

        with torch.no_grad():
            self.diffusion_model.eval()
            vt_pred_con = self.diffusion_model(
                xt, 
                t, 
                semantic_context=condition_tokens.detach(),
                semantic_force_cfg=0,
            )
            vt_pred_uncon = self.diffusion_model(
                xt, 
                t, 
                semantic_context=condition_tokens.detach(),
                semantic_force_cfg=1,
            )
            # consistency model
            et_pred_con = (alphas * vt_pred_con + deltas * xt) 
            et_pred_uncon = (alphas * vt_pred_uncon + deltas * xt) 

            # guidance scale
            lower_bound = 1.5
            upper_bound = 6
            guidance_scale = torch.rand(size=[b], device=self.device, dtype=vocoder_embs.dtype)
            guidance_scale = (lower_bound - upper_bound) * guidance_scale + upper_bound
        
            # t_{n-k}  and t
            # TODO: exclude t_n = 0 ?
            t_n = torch.maximum(t - 0.02, torch.zeros_like(t, device=self.device))
            angles_n = math.pi /2. * t_n
            alphas_n, deltas_n = torch.cos(angles_n), torch.sin(angles_n)
            # ddim solver
            # can we simplify this part?
            ddim_con = (alphas_n / alphas) * xt - deltas_n *((deltas * alphas_n / alphas * deltas_n) - 1) * et_pred_con - xt
            ddim_uncon = (alphas_n / alphas) * xt - deltas_n *((deltas * alphas_n / alphas * deltas_n) - 1) * et_pred_uncon - xt
            xt_n = xt + (1 + guidance_scale[..., None, None]) * ddim_con - guidance_scale[..., None, None] * ddim_uncon

            # TODO: use ema weights to compute c1, verify if the usage is correct
            # TODO: veify if we can do no grad here
            with self.optimizers(use_pl_optimizer=False).swap_ema_weights():
                c1 = self.consistency_model(
                    xt_n.detach(),
                    t_n,
                    semantic_context=condition_tokens.detach(),
                    semantic_force_cfg=0,
                    guidance_scale=guidance_scale
                ).detach()
         
        c0 = self.consistency_model(
            xt,
            t,
            semantic_context=condition_tokens.detach(),
            semantic_force_cfg=0,
            guidance_scale=guidance_scale
        )

        # compute consistency output 
        # c_skip * xt + c_skip * model(xt, c ,t)
        c_skip_0, c_out_0 = self.get_scalings_for_boundary_condition_discrete(t)
        c_skip_1, c_out_1 = self.get_scalings_for_boundary_condition_discrete(t_n)
        c0 = c_skip_0 * xt + c_out_0 * (alphas * xt - deltas * c0)
        c1 = c_skip_1 * xt_n + c_out_1 * (alphas_n * xt_n - deltas_n * c1)

        with torch.autocast(device_type="cuda", enabled=False):
            # unweighted_loss = self.loss_function(vt_pred[:, :, vc_condition_emb.shape[-1]:].float(), vt[:, :, vc_condition_emb.shape[-1]:].float())
            unweighted_loss = self.loss_function(c0.float(), c1.float())
            # rank_zero_info(t[:, 0, 0])
            # rank_zero_info(t_n[:, 0, 0])
            # rank_zero_info(guidance_scale)
            # rank_zero_info(unweighted_loss.mean(dim=-1).mean(dim=-1))
            unweighted_loss = torch.mean(unweighted_loss)
            loss = torch.mean(unweighted_loss)

        self.log_dict(
            {
                "training/loss": loss, 
                "train_loss": loss, 
                "unweighted_loss": unweighted_loss,
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
        os.makedirs(
            f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}", exist_ok=True
        )
        
        self.val_output_dict = {}
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.autocast(device_type="cuda", enabled=False):
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

            # diffusion sampling
            pred_emb = self.sampler(
                model=self.consistency_model, 
                semantic_context=condition_tokens,
                num_items=condition_tokens.shape[0], # batch size: how many samples to generate
                num_chunks=self.hparams.num_chunks,
                num_steps=4, # diffusion steps
                bf16_portion=0.0,
                angle_schedule='uniform',
                schdeule_slope=2.5,
                classifier_free_guidance=2.5,
                vc_context=vc_condition_emb.mean(dim=-1, keepdim=True).permute(0, 2, 1) if self.hparams.vc else None,
                vc_prefix=vc_condition_emb if self.hparams.vc else None,
            )
            # generate audio
            wavs_g = self.vocoder_embs_to_wav(pred_emb.float())

        # save the output wavs
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

        # TODO: FAD?



