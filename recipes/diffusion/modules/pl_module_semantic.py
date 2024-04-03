import os
import math
import torch
import torchaudio
import soundfile as sf
import pytorch_lightning as pl
from einops import rearrange, repeat

from pytorch_lightning.utilities.rank_zero import rank_zero_info
from recipes.diffusion.models.diffusion_sampler import Sampler
from recipes.soundstream.utils.sample_pool import SamplePool
from recipes.bigmusic.lightning.embedding_modules import (
    MulanTagEmbedder, LyricsTokenEmbedder,
)
from recipes.diffusion.utils.utils import random_side_mask

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
        ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=['diffusion_model'])
   
        self.loss_function = torch.nn.MSELoss()
        
        self.requires = {}
        self.model = diffusion_model 
        # load pretrained model
        if pretrained_path is not None:
            self.load_from_pretrained(pretrained_path)
        # lora
        if lora:
            for k, v in self.model.named_parameters():
                if 'lora' not in k and 'vc_null_embedding' not in k and 'vc_context_embed' not in k and 'sep_embedding' not in k:
                    v.requires_grad = False
                else:
                    rank_zero_info(f'Enable gradient for {k}')

        # input embedders
        embedder_dict = {
            'mulan': MulanTagEmbedder(input_dim=512, embedding_dim=1024, add_sos=True),
            'lyrics_tokens': LyricsTokenEmbedder(vocab_size=400, embedding_dim=1024, add_sos=True),
        }
        self.input_embedders = torch.nn.ModuleDict(embedder_dict)

        self.sampler = Sampler(
            in_channels=target_dim,
            duration=duration,
            num_splits=num_chunks,
        )
        self.current_step = 0
        if sample_pool_size > 0:
            self.sample_pool = SamplePool(
                cache_size=sample_pool_size,
                batch_size=train_batch_size,
                length_samples=int(duration*diffusion_sample_rate),
                silence_prob=0.05,
            )
        if diffusion_sample_rate != tokenizer_sample_rate:
            self.resampler = torchaudio.transforms.Resample(diffusion_sample_rate, tokenizer_sample_rate)
    
    def load_from_pretrained(self, pretrained_path=None):
        rank_zero_info('Loading pre-trained model from checkpoint', pretrained_path)
        ckpt_state_dict = torch.load(
            pretrained_path, map_location=torch.device("cpu")
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
        # init required models
        self.load_required_modules()
        # set device for sampler
        self.sampler.set_device(torch.device(f"cuda:{self.local_rank}"))

        # set torch seed for randomness
        torch.manual_seed(self.hparams.seed + self.global_rank)
        if getattr(self.hparams, "resampler", False):
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
        base_params, no_decay_params = self._divide_params_group(self.model)

        optimizer = self.hparams.optimizer_cls(
            [{"params": base_params}, {"params": no_decay_params, "weight_decay": 0.0}],
        )
        scheduler = self.hparams.scheduler_cls(optimizer)
        return [optimizer], [scheduler]
    
    def infer_batch_size(self, batch):
        batch_size = [len(t) for t in batch.values() if torch.is_tensor(t) or isinstance(t, list)][0]
        return batch_size
    
    def prepare_inputs_embeddings(self, batch):
        conditions = batch['conditions'].split(',')
        batch_size = self.infer_batch_size(batch)
        with_sos=True
        # convert inputs to conditions
        inputs_embeds = []
        if 'style_text' in conditions:
            embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_text'], with_sos=with_sos, data_type='text')
            inputs_embeds.append(embeds)
        elif 'style_audio' in conditions:
            embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_audio'].to(self.device), with_sos=with_sos, data_type='music')
            inputs_embeds.append(embeds)
        elif 'style_tag' in conditions: # using Mulan for on-the-fly MIR tagging
            embeds = self.input_embedders['mulan'].embed(self.requires, batch['style_audio'].to(self.device), with_sos=with_sos, data_type='tag')
            inputs_embeds.append(embeds)
        else:
            # adding SOS token no matter what so that all parameters get used
            inputs_embeds.append(self.input_embedders['mulan'].get_sos_embed(batch_size))
        if 'lyrics_tokens' in conditions:
            embeds = self.input_embedders['lyrics_tokens'].embed(self.requires, batch['lyrics_tokens'].to(self.device), with_sos=with_sos)
            inputs_embeds.append(embeds)
        else:
            inputs_embeds.append(self.input_embedders['lyrics_tokens'].get_sos_embed(batch_size))
        return torch.cat(inputs_embeds, dim=1)

    @torch.no_grad()
    def get_semantic_embs(self, x):
        lit_module = self.requires['Stage3']
        encoded_feature = lit_module.model.wav2embed(x)
        shared_encoder_output = lit_module.model.shared_encoder.forward_to_vq(
            encoded_feature, vq = lit_module.model.vq
        )
        vq_emb = shared_encoder_output["vq_emb"]
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
        
        # get condition signals
        with torch.autocast(device_type="cuda", enabled=False):
            if self.hparams.vc:
                vc_condition_emb = self.get_vc_condition_emb(batch['vocal_audio'].float())
            
            semantic_embs = self.get_semantic_embs(batch['audio'].float())
            semantic_embs = rearrange(semantic_embs, 'b l d -> b d l')
            condition_embeds = self.prepare_inputs_embeddings(batch)
           
        # get training target
        b, d, l = semantic_embs.shape
        et = torch.randn_like(semantic_embs)

        # define noise level 
        lower_bound = self.hparams.noise_range[0]
        upper_bound = self.hparams.noise_range[1]
        t = torch.rand(size=[b, 1, self.hparams.num_chunks], device=self.device, dtype=semantic_embs.dtype)
        t = (lower_bound - upper_bound) * t + upper_bound
        # extent t to l, use clone to avoid call by reference
        t = repeat(t, 'b 1 n -> b 1 (n l)', l=l).clone().detach()

        # multi task
        task_dice = torch.randint(0, 2, (1,)).item()
        is_causal = False
        # generation
        if task_dice == 0:
            # causal or non-causal
            causal_dice = torch.randint(0, 2, (1,)).item()
            if causal_dice == 0:
                is_causal = True
        # inpainting / outpainting
        elif task_dice == 1:
            t = random_side_mask(t, min_mask_percentage=0.2, max_mask_percentage=0.8)

        # WIP causal continuation, need to implement mask
        elif task_dice == 2:
            pass

        # diffusion input and output
        angles = math.pi /2. * t
        alphas, deltas = torch.cos(angles), torch.sin(angles)
    
        xt = alphas * semantic_embs + deltas * et
        vt = alphas * et - deltas * semantic_embs

        # diffusion model
        vt_pred = self.model(
            xt, 
            t, 
            semantic_context=condition_embeds,
            vc_context=vc_condition_emb if self.hparams.vc else None,
            is_causal=is_causal,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            unweighted_loss = self.loss_function(vt_pred.float(), vt.float())
            loss = torch.mean(unweighted_loss)

        self.log_dict(
            {
                "training/loss": loss, 
                "train_loss": loss, 
                "unweighted_loss": torch.mean(unweighted_loss),
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

            if self.hparams.vc:
                vc_condition_emb = self.get_vc_condition_emb(batch['vocal_audio'][:self.hparams.val_batch_size].float())
    
            condition_tokens = self.get_condition_tokens(condition_audio)

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
                vc_context=vc_condition_emb if self.hparams.vc else None,
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



