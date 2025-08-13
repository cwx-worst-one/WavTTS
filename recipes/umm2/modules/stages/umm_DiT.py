import torch, os
import pytorch_lightning as pl
from recipes.umm2.modules.stages.stage2 import Stage2
from torch.nn.utils.rnn import pad_sequence
import math
import torch.nn.functional as F
from samantha.utils.hparams import DotDict
from samantha.utils.utils import download_checkpoint
from recipes.sacodec.modules.sacodec_module_umm import SACodecModule as SACodecModuleUMM
from recipes.sacodec.modules.sacodec_module import SACodecModule
from recipes.musiclm.utils.dist import local_zero_first
from recipes.umm2.modules.utils import get_quant_rate

from recipes.voicebox.modules.loss import sequence_mask, MaskedMAELoss, MaskedBCELoss, MaskedSSIMLoss, MaskedMSELoss

LOSS_DICT = {
    "l1": MaskedMAELoss,
    "l2": MaskedMSELoss,
    "ssim": MaskedSSIMLoss
    }

def init_sacodec(checkpoint_path, local_rank, cache_dir=None, version="umm"):
    with local_zero_first():
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        device = torch.device(f"cuda:{local_rank}")
        local_path = download_checkpoint(checkpoint_path, cache_dir)
        if version == "umm":
            module_cls = SACodecModuleUMM
        elif version == "conv":
            module_cls = SACodecModule
        else:
            raise Exception("sacodec version not handled")
    
        # vocoder_model = vocoder_model_pl.generator.eval().to(device)
        sacodec_model = module_cls.load_from_checkpoint(
            checkpoint_path=local_path,
            strict=False
        ).eval().to(device)
        sacodec_model.setup('predict')

        return {
            "sacodec": sacodec_model, 
        }

class Stage2DiT(Stage2):
    def __init__(
        self,
        config,
        dit_config,
        bn_config,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        criterions,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):  
        super().__init__(
            config=config,
            model_cls=model_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
    
        self.criterion_dict = {}
        for criterion in criterions:
            self.criterion_dict[criterion] = LOSS_DICT[criterion]()

        # self.umm_dropout = umm_dropout
        # self.umm_pad = umm_pad
        self.bn_config = bn_config 
        self.dit_config = dit_config

    
    def configure_optimizers(self):
        params = self.model.parameters()
        # optimizer_params = []
        # for name, param in self.model.named_parameters():
        #     if name.startswith("DiT") or name.startswith("vq_proj_in"):
        #         optimizer_params.append(param)
        # import pdb; pdb.set_trace()
        optimizer = self.optimizer_cls(params)
        scheduler = self.scheduler_cls(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        } 
    
    @torch.no_grad()
    def get_sacodec_embedding(self, wav):
        # wav = B x CH x L
        model: SACodecModuleUMM = self.requires["sacodec"]
        latents = model.get_latents(wav) # B L D
        latents = model.normalize_features(latents, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std'])
        return latents
    
    @torch.no_grad()
    def sacodec_embs_to_wav(self, latents, overlap_len=1):
        model: SACodecModuleUMM = self.requires["sacodec"]
        if isinstance(latents, list):
            latents = [model.denormalize_features(latent, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std']).transpose(1, 2)
                       for latent in latents]# B D L -> B L D
            audio_hat = [model.decode_latents(latent) for latent in latents]

            if overlap_len == 0:
                return torch.cat(audio_hat, -1)
            overlap_wav_len = math.ceil(overlap_len * self.bn_config["sample_rate"])
            chunks = None
            for chunk_idx in range(len(audio_hat)):
                x = audio_hat[chunk_idx]
                if chunk_idx == 0:
                    x[..., -overlap_wav_len:] *= torch.linspace(1, 0, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device)  # fade out only
                    chunks = x
                else:
                    if chunk_idx == len(audio_hat) - 1:
                        # x = x[..., :-pad_wav_len] # remove padding
                        x[..., :overlap_wav_len] *= torch.linspace(0, 1, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device) # fade in only
                    else:
                        # fade in & out
                        x[..., :overlap_wav_len] *= torch.linspace(0, 1, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device)
                        x[..., -overlap_wav_len:] *= torch.linspace(1, 0, overlap_wav_len).unsqueeze(0).unsqueeze(0).to(x.device)
                    chunks = torch.cat([chunks[..., :-overlap_wav_len], 
                                        chunks[..., -overlap_wav_len:] + x[..., :overlap_wav_len],
                                        x[..., overlap_wav_len:]
                                        ], dim=-1) 
                print(f"chunk {chunk_idx} shape: ", chunks.shape)
            print(chunks.shape)
            return chunks

        else:
            latents = model.denormalize_features(latents, self.bn_config["bn_norm_mean"], self.bn_config['bn_norm_std'])
            latents = latents.transpose(1, 2) # B D L -> B L D
            audio_hat = model.decode_latents(latents)
        
        return audio_hat # B x CH x L

    @torch.no_grad()
    def get_bn(self, batch, wav_key="audio_44100", wav_len_key="wav_lens"):
        batch["bn"] = self.get_sacodec_embedding(batch[wav_key]) # B x L x D
        # sacodec_lens = (batch["audio_length"][-1] / 44100 * self.bn_config["bn_frame_rate"]).round() # TODO: replace magic numbers with config values
        sacodec_lens = (batch[wav_len_key] / self.bn_config["sample_rate"] * self.bn_config["bn_frame_rate"]).round() 
        # print('Sacodec feat', batch["bn"].shape, sacodec_lens)
        batch["bn_lens"] = sacodec_lens
        batch["bn_mask"] = sequence_mask(sacodec_lens, max_len=batch["bn"].shape[1], device=sacodec_lens.device)
        # print('UMM', batch["token"].shape)
        return batch
    
    @staticmethod
    def get_nframe(audio_len, nhop):
        return math.ceil(audio_len / nhop)
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_msr_wav(self, batch):
        if isinstance(batch, list):
            new_batch = {
                "wav_24k": pad_sequence([item["wav_24k"][0] for item in batch], batch_first=True, ),
                "wav": pad_sequence([item["wav"][0].mT for item in batch], batch_first=True).mT,
                "wav_lens": torch.tensor([item["wav"].size(-1) for item in batch]).long(), 
            }
            new_batch["wav_lens"] = new_batch["wav_lens"].to(new_batch["wav"].device)
            batch = new_batch
        
        if "wav_24k" in batch and "wav" in batch:
            wav_24k_mono, wav_44k1_stereo = batch["wav_24k"], batch["wav"]
            wav_len_key = "wav_lens"
        elif "audio_24000" in batch and "audio_44100" in batch:
            wav_24k_mono, wav_44k1_stereo = batch["audio_24000"], batch["audio_44100"]
            wav_len_key = "audio_44100_length"
        else:
            raise ValueError(f"wav_24k_mono and wav_44k1_stereo not found in batch, {batch.keys()}")

        if wav_24k_mono.dim() == 3:
            wav_24k_mono = wav_24k_mono[:, 0]   # stereo 24k -> mono 24k

        sample_rate1, sample_rate2 = self.config.sample_rate, self.bn_config["sample_rate"]
        frame_rate1, frame_rate2 = self.config.frame_rate, self.bn_config["bn_frame_rate"]
        rate1 = int(sample_rate1 / frame_rate1)

        hop_length1, hop_length2 = self.config.hop_length, self.bn_config["hop_size"]
        mel_frame_rate = sample_rate1 // hop_length1    # 100
        if wav_24k_mono.size(-1) % rate1 > 0:
            wav_pad_len = rate1 - (wav_24k_mono.size(-1) % rate1)
            wav_24k_mono = F.pad(wav_24k_mono, (0, wav_pad_len), "constant", 0)

        frame1 = self.get_nframe(wav_24k_mono.size(-1), hop_length1)     # mel frame_rate: 100
        # target_frame2 = frame1 // 2                       # sacodec frame_rate: 50
        target_wav2_len = round(frame1 / mel_frame_rate * sample_rate2)
        if target_wav2_len > wav_44k1_stereo.size(-1):
            wav_44k1_stereo = F.pad(wav_44k1_stereo, (0, target_wav2_len - wav_44k1_stereo.size(-1)), "constant", 0)
        else:
            wav_44k1_stereo = wav_44k1_stereo[..., :target_wav2_len]
        # frame2 = get_nframe(wav2.size(-1), hop_length2) 
        frame2 = round(wav_44k1_stereo.size(-1) / sample_rate2 * self.bn_config["bn_frame_rate"])

        if frame1 / mel_frame_rate != frame2 / self.bn_config["bn_frame_rate"]:
            wav_24k_mono = F.pad(wav_24k_mono, (0, int(60*sample_rate1) - wav_24k_mono.size(-1)), "constant", 0)
            wav_44k1_stereo = F.pad(wav_44k1_stereo, (0, int(60*sample_rate2) - wav_44k1_stereo.size(-1)), "constant", 0)
        
        batch["audio"] = wav_24k_mono
        batch["audio_length"] = batch["audio_24000_length"] # required by rmpad version
        batch["audio_44100"] = wav_44k1_stereo
        return batch, wav_len_key


    def _shared_step(self, batch):
        batch, wav_len_key = self.prepare_msr_wav(batch)
        batch = self.get_bn(batch, wav_len_key=wav_len_key)
        ref = batch["bn"]
        loss_mask = batch["bn_mask"]
        feat_len = batch["bn_lens"]

        bsz, seqlen = ref.shape[0], ref.shape[1]
        batch_tokens = torch.sum(feat_len).item()

        loss_dict = {
            "num_tokens": batch_tokens,
        }

        # torch.save(batch, "batch.pt")
        output_dict = self.model(batch)

        mel = output_dict["mel"]
        loss_dict["bs"] = mel.shape[0]
        # loss_dict["flops"] = output_dict["flops"]
        
        if "text_ids" in output_dict:
            text_ids = output_dict["text_ids"]
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)

        for key in output_dict:
            if key.startswith("loss"):
                loss_dict.update({key: output_dict[key]})

        # diffusion loss
        dit_loss = 0
        for loss_type, loss_func in self.criterion_dict.items():
            tmp_loss = loss_func(output_dict["dit_pred"], output_dict["dit_target"], loss_mask)
            loss_dict["dit_"+loss_type] = tmp_loss.item()
            dit_loss += tmp_loss
        loss_dict["dit_loss"] = dit_loss.item()

        # rvq & vq
        if "loss_rvq" in output_dict and output_dict["loss_rvq"] is not None:
            # code rate & quant rate
            for r in range(output_dict["vq_ids"].shape[-1]):
                quant_rate = get_quant_rate(self,
                    output_dict["vq_ids"][...,r].long(), self.config.vq_codebook_size
                )
                loss_dict[f"aux/quant_rate{r}"] = quant_rate
                if "vq_entropy" in output_dict:
                    loss_dict[f"aux/entropy{r}"] = output_dict["vq_entropy"][..., r]
                if "ppl" in output_dict:
                    loss_dict[f"aux/ppl{r}"] = output_dict["ppl"][..., r]

                loss_dict[f"loss_rvq{r}"] = output_dict["loss_rvq"][..., r]
            loss_dict["loss_rvq"] = loss_dict["loss_rvq"].sum()
        
        if "loss" in output_dict:
            loss_dict["loss"] = dit_loss + output_dict["loss"]
        else:
            loss_dict["loss"] = dit_loss

        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        
        return loss_dict
