import logging
import os
import random
import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import matplotlib.pyplot as plt
import torch.nn as nn
from recipes.diffusion.utils.utils import download_checkpoint
from recipes.musiclm.utils.dist import local_zero_first
from recipes.music_dit.model.loss import MaskedMAELoss, MaskedSSIMLoss, MaskedMSELoss, MaskedMAELossDim2, sequence_mask
from recipes.music_dit.utils.infer_utils import save_wav
from recipes.umm.requires.model_initializer import init_stage3_dual_voc
from samantha.utils.hparams import DotDict
# TODO (qq) creat embedding_modules.py in music_dit directory.
from recipes.bigmusic.lightning.embedding_modules import (
    MultiTagsCategoricalEmbedder,
    SpeakerEmbedder,
    LyricsTokenEmbedder,
)

logger = logging.getLogger(__name__)

LOSS_DICT = {
    'l1': MaskedMAELoss,
    'l1dim2': MaskedMAELossDim2,
    'l2': MaskedMSELoss,
    'ssim': MaskedSSIMLoss
}


def fix_flashattn_version(model_cls):
    hp = model_cls.keywords['hp']
    if hp.use_window_mask and hp.flashattn_version != '2.3':
        print('WARN: try to change flashattn_version from 2 to 2.3 when use_window_mask=True')
        hp.flashattn_version = '2.3'
        model_cls.keywords['hp'] = hp
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
        val_output_samples_dir='',
        extra_params=None,
    ):
        super().__init__()
        self.extra_params = DotDict(extra_params)
        self.save_hyperparameters()
        # self.model = fix_flashattn_version(model_cls)()
        self.model = model_cls()

        self.input_embedders = {}
        prefix_hidden_size = extra_params['prefix_hidden_size']
        # TODO: temporal_aligned_hidden_size = extra_params['temporal_aligned_hidden_size']
        default_input_embedders = {
            'temporal_aligned': [], 
            'prefix': ['multitags_categorical', 'speaker_id', 'lyrics_tokens'],
            'xattn': []}
        for input_type, emb_types in extra_params.get('input_embedders', default_input_embedders).items():            
            embedder_dict = {}
            for emb_type in emb_types:
                if emb_type == 'multitags_categorical':
                    embedder_dict[emb_type] = MultiTagsCategoricalEmbedder(
                        embedding_dim=prefix_hidden_size,
                        add_sos=True,
                        dropout=0.1,
                        vocab_type=extra_params['tag_taxonomy_lang'],
                    )
                elif emb_type == 'speaker_id':
                    embedder_dict[emb_type] = SpeakerEmbedder(
                        vocab_size=extra_params['speaker_codebook_size'], 
                        embedding_dim=prefix_hidden_size, 
                        add_sos=True,                        
                    )
                elif emb_type == 'lyrics_tokens':
                    embedder_dict[emb_type] = LyricsTokenEmbedder(
                        vocab_size=extra_params['lyrics_codebook_size'],
                        embedding_dim=prefix_hidden_size,
                        add_sos=True,
                        add_eos=False,
                    )
            self.input_embedders[input_type] = nn.ModuleDict(embedder_dict)

        self.null_embs = torch.nn.Parameter(torch.randn(prefix_hidden_size))
        
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
            local_path = download_checkpoint(
                pretrained_path, 
                cache_dir=self.extra_params.cache_dir,
            )

            ckpt_state_dict = torch.load(
                local_path, map_location=torch.device('cpu')
            )['state_dict']
            model_state_dict = self.model.state_dict()
            new_state_dict = {}

            for k in ckpt_state_dict:
                new_k = k.replace('model.', '') # saved model has prefix 'model.'
                if new_k in model_state_dict:
                    if ckpt_state_dict[k].shape != model_state_dict[new_k].shape:
                        rank_zero_info(f'Skip loading parameter: {k}, '
                                    f'required shape: {model_state_dict[new_k].shape}, '
                                    f'loaded shape: {ckpt_state_dict[k].shape}')
                    else:
                        new_state_dict[new_k] = ckpt_state_dict[k]
                else:
                    rank_zero_info(f'Dropping parameter {k}')

            self.model.load_state_dict(new_state_dict, strict=False)

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    def infer_batch_size(self, batch):
        batch_size = [len(t) for t in batch.values() if torch.is_tensor(t) or isinstance(t, list)][0]
        return batch_size

    def infer_conditions(self, batch):
        if type(batch['conditions']) == list:
            assert (
                len(set(list(map(tuple, batch['conditions'])))) == 1
            ), 'Make sure that all conditions in the batch are the same'
            conditions = batch['conditions'][0].split(',')
        else:
            conditions = batch['conditions'].split(',')
        return conditions

    def prepare_audio_inputs(self, batch):
        assert 'target_audio' in batch
        batch['cond_audio'] = batch.get('cond_audio', None)                
        if 'melae' in self.requires:        # melae
            melae = self.requires['melae']
            batch['bn'] = melae.wav2token(batch['target_audio'])
            if batch['cond_audio']:
                batch['cond_bn'] = melae.wav2token(batch['cond_audio'])
        elif 'vocoder' in self.requires:    # music vae
            wvae = self.requires['vocoder']            
            h_ = wvae.encode(batch['target_audio'][:, None])
            batch['bn'], _, _ = wvae.sample(h_, deterministic=False)
            batch['bn'] = batch['bn'].transpose(1, 2)
            if batch['cond_audio']:
                h_ = wvae.encode(batch['cond_audio'][:, None])
                batch['cond_bn'], _, _ = wvae.sample(h_, deterministic=False)
                batch['cond_bn'] = batch['cond_bn'].transpose(1, 2)
            batch['seqlen'] = batch['seqlen'] * 10  # 12.5hz to 125hz
        return batch

    def prepare_temporal_aligned_inputs(self, batch):        
        return batch

    def prepare_xattn_inputs(self, batch):        
        return batch        

    def prepare_prefix_inputs(self, batch):
        batch_size = self.infer_batch_size(batch)
        conditions = self.infer_conditions(batch)
        inputs_embeds = []
        prefix_embedders = self.input_embedders['prefix']
        for emb_type, embedder in prefix_embedders.items():
            if emb_type == 'multitags_categorical':
                if 'style_text' in conditions:
                    embeds = embedder.embed(self.requires, batch['style_text'], with_sos=True)
                else:
                    embeds = embedder.get_sos_embed(batch_size)
            if emb_type == 'speaker_id':
                if 'speaker_id' in conditions:
                    # TODO: fix speaker_id when shape=1
                    if len(batch['speaker_id'].shape) == 1:
                        batch['speaker_id'] = batch['speaker_id'].view(-1, 1)
                    embeds = embedder.embed(self.requires, batch['speaker_id'].cpu(), with_sos=False)
                else:
                    embeds = embedder.get_sos_embed(batch_size)
            if emb_type == 'lyrics_tokens':
                if 'lyrics_tokens' in conditions:
                    embeds = embedder.embed(self.requires, batch['lyrics_tokens'].cpu(), with_sos=True)
                else:
                    embeds = embedder.get_sos_embed(batch_size)
            inputs_embeds.append(embeds)
        inputs_embeds = torch.cat(inputs_embeds, dim=1).to(self.device)
        # TODO (qq) Apply separate dropout to different prefix embeds. 
        if self.training:
            mask_embed = torch.unsqueeze(torch.unsqueeze(self.null_embs, 0), 0)
            mask_embed = mask_embed.repeat(inputs_embeds.shape[0], inputs_embeds.shape[1], 1)
            masked_b = torch.rand_like(mask_embed[:, 0, 0].float())
            masked_b = (masked_b[:, None, None] < 0.15).long()
            inputs_embeds = inputs_embeds * (1 - masked_b) + mask_embed * masked_b
        batch['prefix_inputs_emb'] = inputs_embeds
        return batch

    def prepare_input(self, batch, Tctx=None):
        # Audio inputs: batch['bn'], batch['cond_bn'], batch['seqlen'] 
        batch = self.prepare_audio_inputs(batch)
        # Temporal aligned inputs: batch['temporal_aligned_inputs_emb']
        batch = self.prepare_temporal_aligned_inputs(batch)
        # Prefix inputs: batch['prefix_inputs_emb']
        batch = self.prepare_prefix_inputs(batch)
        # Xattn inputs: batch['xattn_inputs_emb']
        batch = self.prepare_xattn_inputs(batch)

        # Add bn mask
        Tmax = batch['bn'].shape[1]
        batch['seqlen'] = batch['seqlen'].clamp_max(Tmax)

        # TODO (qq) use a class for bn_ctx_mask, support inpainting.
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
                'loss': loss.item(),
                'bsz': bsz,
                'seqlen': seqlen
            }
            log_dict.update(loss_dict)
            log_dict['training/loss'] = log_dict['loss']
            self.log_dict(log_dict, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            },
        }

    @torch.no_grad()
    def inference(self, inputs, step):
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        x = self.model.inference(inputs, step)
        return x

    predict = inference

    def latent2wav(self, latents):
        if 'melae' in self.requires:
            mel = self.requires['melae'].latent2mel(latents)
            wav = self.melvoc(mel.transpose(1, 2), None).reshape(-1)
        else:
            wvae = self.requires['vocoder']
            wav = wvae.decode(latents.float().transpose(1, 2)).reshape(-1)
        return wav

    def validation_step(self, inputs, step):
        bsz, seqlen, loss_mask = self.prepare_input(
            inputs, Tctx=0)
        # inputs, strip_txt_padding=True, Tctx=inputs["seqlen"][0] // 3)
        diffusion_nfe = self.extra_params.diffusion_nfe
        diffusion_sampler = self.extra_params.diffusion_sampler
        text_cfg_w = self.extra_params.text_cfg_w
        batch_num = inputs["bn"].shape[0]
        assert batch_num == 1

        inputs['bn_ctx'] = inputs['bn_ctx'].repeat(2, 1, 1)
        inputs['bn_ctx_mask'] = inputs['bn_ctx_mask'].repeat(2, 1, 1)
        
        if 'prefix_inputs_emb' in inputs:
            inputs_embeds = inputs['prefix_inputs_emb']
            cfg_prefix_inputs_emb = torch.unsqueeze(torch.unsqueeze(self.null_embs, 0), 0)     
            cfg_prefix_inputs_emb = cfg_prefix_inputs_emb.repeat(inputs_embeds.shape[0], inputs_embeds.shape[1], 1)
            inputs['prefix_inputs_emb'] = torch.concat([inputs_embeds, cfg_prefix_inputs_emb], dim=0)

        # TODO (qq) add cfg input for temporal_aligned_inputs_emb and xattn_inputs_emb
        
        lat = self.model.inference(
            inputs=inputs,
            timesteps=diffusion_nfe,
            sampler=diffusion_sampler,
            text_cfg_w=text_cfg_w)
        generated_audio = self.latent2wav(lat.transpose(1, 2))
        outputs = {
            "generated_audio":generated_audio.reshape(batch_num, -1),
            "generated_audio_tensor":generated_audio.reshape(1, -1),
        }
        if 'target_audio' in inputs:
            outputs.update({"target_audio":inputs['target_audio'].float().cpu().numpy().reshape(batch_num, -1)})
        return outputs

    def prepare_latent_vocoder(self):
        with local_zero_first():
            local_path = download_checkpoint('hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/renyi/samantha_ckpts/voc/v1/checkpoints/vocoder_last.ckpt', '.module_cache')
            self.melvoc = init_stage3_dual_voc(
                str(local_path),
                cache_dir='.module_cache', local_rank=self.local_rank)['mel_vocoder']

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.prepare_latent_vocoder()

    def on_predict_epoch_start(self) -> None:
        super().on_predict_epoch_start()
        self.prepare_latent_vocoder()

    def predict_step(self, batch, batch_idx=0, dataloader_idx=0):
        return self.validation_step(batch, step=batch_idx)


class Lyric2songInferenceModule(VoiceBoxModule):
    def prepare_audio_inputs(self, batch):
        seqlen = math.ceil(self.extra_params.semantic_frame_rate * self.extra_params.duration)
        batch['bn'] = torch.zeros([1, seqlen, self.extra_params.latent_dims]).to(self.device)
        batch['seqlen'] = torch.LongTensor([seqlen]).to(self.device)

        batch['cond_audio'] = batch.get('cond_audio', None)   
        if 'melae' in self.requires:        # melae
            melae = self.requires['melae']
            if batch['cond_audio']:
                batch['cond_bn'] = melae.wav2token(batch['cond_audio'])
        elif 'vocoder' in self.requires:    # music vae
            wvae = self.requires['vocoder']   
            if batch['cond_audio']:
                h_ = wvae.encode(batch['cond_audio'][:, None])
                batch['cond_bn'], _, _ = wvae.sample(h_, deterministic=False)
                batch['cond_bn'] = batch['cond_bn'].transpose(1, 2)
        return batch