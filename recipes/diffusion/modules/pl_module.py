import os
import math
import torch
import soundfile as sf
import pytorch_lightning as pl
from collections import OrderedDict
from einops import rearrange, repeat
from recipes.diffusion.models.semantic_model.utils import (
    init_wav2vec, 
    init_semantic_centers,
    w2v_bert_tokenization
)
from recipes.diffusion.models.mulan_model.utils import init_mulan, init_mulan_centers, mulan_inference_wrapper
from recipes.diffusion.models.vocoder_model.utils import init_vocoder, init_vocoder_yongye
from recipes.diffusion.models.dualpath_net import DualPathDiffusionNetwork
from recipes.diffusion.models.tnt import TNTDiffusionNetwork
from recipes.diffusion.models.diffusion import ARVSampler

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DiffusionModule(pl.LightningModule):
    def __init__(
            self, 
            seed,
            noise_range,
            sample_rate,
            asset_dir,
            model_name, 
            optimizer_cls, 
            scheduler_cls,
            mulan_model,
            mulan_centers,
            semantic_model,
            semantic_centers,
            vocoder_model,
            val_output_samples_dir,
        ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters()
        # self.model = get_model(self.hparams.model_name)
        self.num_chunks = 1
        self.chunk_length = 2500
        self.loss_function = torch.nn.MSELoss()

        self.model = TNTDiffusionNetwork(
            input_dim=16,
            feature_dim=1024,
            context_dim=1,
            depth=16,
            num_chunks=4,
            segment_size=64,
            segment_stride=64,
            dropout=0,
            mulan_cfg_prob=0.4,
            semantic_cfg_prob=0.4,
            use_checkpoint=False
        )
        # path = 'diffusion-step=030999.ckpt'
        # ckpt = torch.load(path, map_location='cpu')
        # new_dict = OrderedDict()
        # replace_list = ['null_emb', 'fine_rotary_emb', 'coarse_rotary_emb']
        # for key in ckpt['state_dict']:
        #     if 'model' in key:
        #         new_key = key.replace('model.', '')
            #     for r in replace_list:
            #         if r in new_key:
            #             new_key = new_key.replace('emb', 'embedding')

        #     new_dict[new_key] = ckpt['state_dict'][key]
        # self.model.load_state_dict(new_dict)

        self.sampler = ARVSampler(
            in_channels=16,
            length=self.num_chunks*self.chunk_length,
            num_splits=self.num_chunks,
        )
        
        # custom recorder for training step due to GAN training
        self.current_step = 0

    
    def on_fit_start(self):
        # init required models
        self.mulan_model = init_mulan(
            trainer=self.trainer,
            path=self.hparams.mulan_model['model_path'],
            device=self.device,
            cache_dir=self.hparams.asset_dir,
        )
        self.mulan_centers = init_mulan_centers(
            trainer=self.trainer,
            path=self.hparams.mulan_centers['model_path'],
            device=self.device,
            cache_dir=self.hparams.asset_dir,
        )
        self.semantic_model = init_wav2vec(
            trainer=self.trainer,
            path=self.hparams.semantic_model['model_path'],
            device=self.device,
            cache_dir=self.hparams.asset_dir,
        )
        self.vocoder_model = init_vocoder_yongye(
            trainer=self.trainer,
            path=self.hparams.vocoder_model['model_path'],
            device=self.device,
            cache_dir=self.hparams.asset_dir,
        )
        self.semantic_centers = init_semantic_centers(
            trainer=self.trainer,
            path=self.hparams.semantic_centers['model_path'],
            device=self.device,
            cache_dir=self.hparams.asset_dir,
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
    def get_mulan_tokens(self, x, domain='audio'):
        if domain == 'audio':
            mulan_tokens = mulan_inference_wrapper(
                self.mulan_model['mulan_model'], 
                self.mulan_centers['mulan_centers'],
                music=x,
                text=None, 
                device=self.mulan_model['mulan_model'].device
            )
        elif domain == 'text':
            mulan_tokens = mulan_inference_wrapper(
                self.mulan_model['mulan_model'], 
                self.mulan_centers['mulan_centers'],
                music=None,
                text=x, 
                device=self.mulan_model['mulan_model'].device
            )
        return mulan_tokens

    @torch.no_grad()
    def get_semantic_embs(self, x):
        # input x has shape (b, c, t)
        self.semantic_model["ssl_frontend"].eval()
        self.semantic_model["semantic"].eval()
        b, _, t = x.size()
        feats, feat_mask = self.semantic_model["ssl_frontend"](
            x[:, 0], torch.LongTensor([t]).repeat([b]).to(x.device)
        )
        w2v_embeds, _ = self.semantic_model["semantic"](feats, feat_mask)

        return w2v_embeds.detach()

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
    def get_vocoder_embs(self, x):
        # input x has shape (b, c, t)
        self.vocoder_model["model"].eval()
        encoder_out = self.vocoder_model["model"].encode(x)
        sample, _, _ = self.vocoder_model["model"].sample(encoder_out,  deterministic=False) # TODO: check wethear deterministic should be True or False
        return sample.detach()

    @torch.no_grad()
    def vocoder_embs_to_wav(self, x):
        # input x has shape (b, c, t)
        self.vocoder_model["model"].eval()
        wav = self.vocoder_model["model"].decode(x)
        return wav.detach()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # temp treatment for mcc
        if type(batch) is list:
            batch = {'audio': batch[0]}

        with torch.autocast(device_type="cuda", enabled=False):
            # context
            mulan_tokens = self.get_mulan_tokens(batch['audio'][:, 0].float(), domain='audio')
            semantic_tokens = self.get_semantic_tokens(batch['audio'].float())
            # target
            vocoder_embs = self.get_vocoder_embs(batch['audio'].float())
        
            # diffusion training
            b, d, l = vocoder_embs.shape
            et = torch.randn_like(vocoder_embs)
            # noise level ensemble
            lower_bound = self.hparams.noise_range[0]
            upper_bound = self.hparams.noise_range[1]
            t = torch.rand(size=[b, 1, self.num_chunks], device=self.device, dtype=vocoder_embs.dtype)
            t = (lower_bound - upper_bound) * t + upper_bound
        
            t = repeat(t, 'b 1 n -> b 1 (n l)', l=self.chunk_length)

            angles = math.pi /2. * t
            alphas, deltas = torch.cos(angles), torch.sin(angles)
        
            xt = alphas * vocoder_embs + deltas * et
            vt = alphas * et - deltas * vocoder_embs
        
        vt_pred = self.model(
            xt, 
            t, 
            mulan_context=mulan_tokens,
            semantic_context=semantic_tokens,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            # min-SNR-gamma weighting
            # snr = (alphas/(deltas + 1e-8))**2
            # snr_mg = torch.min(snr, torch.ones_like(snr) * 5)
            # w = snr_mg / (snr + 1)
            # w = repeat(w, 'b 1 n -> b d n', d=xt.shape[1])
            unweighted_loss = self.loss_function(vt_pred.float(), vt.float())
            # loss = torch.mean(w*unweighted_loss)
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
        self.val_output_dict = {}
        return

    def validation_step(self, batch, batch_idx):
        with torch.autocast(device_type="cuda", enabled=False):
            # context
            semantic_tokens, prompts = batch
            mulan_tokens = self.get_mulan_tokens(
                list(prompts), 
                domain='text'
            )

            # diffusion sampling
            pred_emb = self.sampler(
                model=self.model, 
                mulan_context=mulan_tokens,
                semantic_context=semantic_tokens,
                num_items=semantic_tokens.shape[0], # batch size: how many samples to generate
                num_chunks=self.num_chunks,
                num_steps=20, # diffusion steps
                start=None,
                show_progress=False,
                angle_schedule='linear',
                classifier_free_guidance=2.5,
            )
            # generate audio
            wavs_g = self.vocoder_embs_to_wav(pred_emb.float())

        # save the output wavs
        for _id, wav in zip(prompts, wavs_g):
            sf.write(
                f"{self.hparams.val_output_samples_dir}/{self.current_step}/{_id}.wav",
                wav.cpu().numpy().T,
                self.hparams.sample_rate,
            )

        # TODO: FAD?



