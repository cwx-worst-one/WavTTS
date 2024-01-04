import os
import math
import torch
import torchaudio
import soundfile as sf
import pytorch_lightning as pl
from collections import OrderedDict
from einops import repeat
from recipes.diffusion.models.semantic_model.utils import (
    init_wav2vec, 
    init_bestrq,
    w2v_bert_tokenization
)
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from recipes.diffusion.models.dualpath_net import DualPathDiffusionNetwork
from recipes.diffusion.models.tnt_mulan_free import TNTDiffusionNetwork
from recipes.diffusion.models.diffusion_mulan_free import ARVSampler

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DiffusionModule(pl.LightningModule):
    def __init__(
            self, 
            seed,
            diffusion_model,
            target_dim,
            num_chunks,
            chunk_length,
            noise_range,
            sample_rate,
            asset_dir,
            model_name, 
            optimizer_cls, 
            scheduler_cls,
            semantic_model,
            vocoder_model,
            val_output_samples_dir,
        ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=['diffusion_model'])
   
        self.loss_function = torch.nn.MSELoss()
        
        self.model = diffusion_model 
        self.sampler = ARVSampler(
            in_channels=target_dim,
            length=num_chunks*chunk_length,
            num_splits=num_chunks,
        )
        # custom recorder for training step due to GAN training
        self.current_step = 0
        # self.resampler = torchaudio.transforms.Resample(44100, sample_rate)
    
    
    def on_fit_start(self):
        # init required models
        self.semantic_model = init_bestrq(
            trainer=self.trainer,
            path=self.hparams.semantic_model['model_path'],
            device=self.device,
            cache_dir=self.hparams.asset_dir,
        )
        self.vocoder_model = init_vocoder(
            checkpoint_path=self.hparams.vocoder_model['model_path'],
            local_rank=self.local_rank,
            cache_dir=self.hparams.asset_dir,
        )
        # set device for sampler
        self.sampler.set_device(self.device)

        # set torch seed for randomness
        torch.manual_seed(self.hparams.seed + self.global_rank)
        # self.resampler = self.resampler.to(self.device)
    
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
    def get_semantic_tokens(self, x):
        self.semantic_model["ssl_frontend"].eval()
        self.semantic_model["semantic"].eval()
        wav2vec_tokens = w2v_bert_tokenization(
            frontend=self.semantic_model["ssl_frontend"],
            w2v_model=self.semantic_model["semantic"],
            wavs=x[:, 0],
            centers=self.semantic_centers["semantic_centers"],
            device=x.device,
        )
        return wav2vec_tokens.detach()

    @torch.no_grad()
    def get_semantic_tokens(self, x):
        lit_module = self.semantic_model['Stage3']
        vq_ids = lit_module.wav2token(x)
        return vq_ids.detach()

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
        # temp treatment for mcc
        if type(batch) is list:
            batch = {'audio': batch[0]}
        elif 'target_audio' in batch:
            batch['audio'] = batch['target_audio'][:, None, :] # bs, seq_len -> bs, ch1, seq_len

        # batch['audio'] = batch['audio'].mean(dim=1, keepdim=True)
        # batch['audio'] = self.resampler(batch['audio'].float())
        

        with torch.autocast(device_type="cuda", enabled=False):
            # context
            semantic_tokens = self.get_semantic_tokens(batch['audio'].float())
            # target
            vocoder_embs = self.get_vocoder_embs(batch['audio'].float())
        
            # diffusion training
            b, d, l = vocoder_embs.shape
            et = torch.randn_like(vocoder_embs)
            # noise level ensemble
            lower_bound = self.hparams.noise_range[0]
            upper_bound = self.hparams.noise_range[1]
            t = torch.rand(size=[b, 1, self.hparams.num_chunks], device=self.device, dtype=vocoder_embs.dtype)
            t = (lower_bound - upper_bound) * t + upper_bound
        
            t = repeat(t, 'b 1 n -> b 1 (n l)', l=l)

            angles = math.pi /2. * t
            alphas, deltas = torch.cos(angles), torch.sin(angles)
        
            xt = alphas * vocoder_embs + deltas * et
            vt = alphas * et - deltas * vocoder_embs
   
        vt_pred = self.model(
            xt, 
            t, 
            semantic_context=semantic_tokens,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            unweighted_loss = self.loss_function(vt_pred.float(), vt.float())
            loss = torch.mean(unweighted_loss)

        self.log_dict(
            {
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
                batch['audio'] = batch['target_audio'][:, None, :] # bs, seq_len -> bs, ch1, seq_len
            
            # batch['audio'] = batch['audio'].mean(dim=1, keepdim=True)
            # batch['audio'] = self.resampler(batch['audio'].float())
            # context
            semantic_tokens = self.get_semantic_tokens(batch['audio'].float())
            if semantic_tokens.shape[1] < self.hparams.chunk_length:
                pad_len = self.hparams.chunk_length - semantic_tokens.shape[1]
                semantic_tokens = torch.cat(
                    [
                        semantic_tokens,
                        torch.ones((semantic_tokens.shape[0], pad_len), device=self.device).long()
                    ], 
                    dim=1
                )

            # diffusion sampling
            pred_emb = self.sampler(
                model=self.model, 
                semantic_context=semantic_tokens,
                num_items=semantic_tokens.shape[0], # batch size: how many samples to generate
                num_chunks=self.hparams.num_chunks,
                num_steps=25, # diffusion steps
                bf16_portion=0.0,
                start=None,
                show_progress=False,
                angle_schedule='linear',
                schdeule_slope=2.5,
                classifier_free_guidance=2.5,
            )
            # generate audio
            wavs_g = self.vocoder_embs_to_wav(pred_emb.float())

        # save the output wavs
        for idx, (gt, wav) in enumerate(zip(batch['audio'], wavs_g)):
            num_files = len(os.listdir(f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}")) // 2
            sf.write(
                f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}/{num_files}.wav",
                wav.cpu().numpy().T,
                self.hparams.sample_rate,
            )
            sf.write(
                f"{self.hparams.val_output_samples_dir}/{self.current_step}/{self.local_rank}/{num_files}_gt.wav",
                gt.cpu().numpy().T,
                self.hparams.sample_rate,
            )

        # TODO: FAD?



