import torch
from torch import nn
from torch.nn import functional as F

from recipes.umm.models.dualumm_encoders import (
    ConvStacksWithDownUpSampling,
    MultiRefTimbreEncoder,
)
from recipes.umm.models.rmvpe import RMVPE
from recipes.umm.models.umm_mkii import (
    ClusteredVectorQuantizer,
    EMAVectorQuantizerEntropy,
    FiniteScalarQuantizer,
    LookupFreeQuantizer,
    Transpose,
    WNConv1d,
    get_vuv,
)
from recipes.umm.models.vq import EMAVectorQuantizer
from recipes.umm.models.wenet.transformer.decoder import TransformerDecoder
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.utils.mel_utils import torch_wav2spec


class DualUMMv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ds = config.get("downsampling", 4)
        self.us = config.get("upsampling", 1)
        self.conv_hidden_size = config.get("conv_hidden_size", 256)
        self.encoder_layer = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        self.init_vocal_branch(config)
        self.init_inst_branch(config)
        self.init_full_branch(config)
        self.chroma_transform = ChromaSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_chroma=config.n_chroma,
            normalized=False,
        )
        self.rmvpe = RMVPE()

    def init_inst_branch(self, config):
        self.init_vq_layers("inst", config)
        head_hidden_size = self.conv_hidden_size
        self.decoder_inst = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        self.audio_encoder_inst = ConvStacksWithDownUpSampling(
            config.hidden_size,
            config.n_mels_tgt,
            config.hidden_size,
            downsampling=self.ds,
            upsampling=self.us,
        )
        self.mel_head_inst = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            config.n_mels_tgt,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.chroma_head_inst = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            config.n_chroma,
            downsampling=self.us,
            upsampling=self.ds,
        )

    def init_vocal_branch(self, config):
        self.init_vq_layers("vocal", config)
        head_hidden_size = self.conv_hidden_size
        self.audio_encoder_vocal = ConvStacksWithDownUpSampling(
            config.hidden_size,
            config.n_mels_tgt,
            config.hidden_size,
            downsampling=self.ds,
            upsampling=self.us,
        )
        self.decoder_vocal = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        self.mel_head_vocal = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            config.n_mels_tgt,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.f0_vuv_head_vocal = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            2,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.asr_aux_decoder_vocal = TransformerDecoder(
            config.vocab_size, config.hidden_size, 4, config.hidden_size * 4, 4
        )
        self.timbre_enc_vocal = None
        if self.config.vocal_ref:
            self.timbre_enc_vocal = MultiRefTimbreEncoder(
                config.hidden_size, config.n_mels_tgt, config.hidden_size
            )

    def init_full_branch(self, config):
        head_hidden_size = self.conv_hidden_size
        self.decoder_in_proj = nn.Conv1d(
            config.hidden_size * 2, config.hidden_size, 3, 1, 1
        )
        self.decoder_full = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        self.mel_head_full = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            config.n_mels_tgt,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.f0_vuv_head_full = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            2,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.asr_aux_decoder_full = TransformerDecoder(
            config.vocab_size, config.hidden_size, 4, config.hidden_size * 4, 4
        )
        self.chroma_head_full = ConvStacksWithDownUpSampling(
            head_hidden_size,
            config.hidden_size,
            config.n_chroma,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.timbre_enc_full = None
        if self.config.vocal_ref:
            self.timbre_enc_full = MultiRefTimbreEncoder(
                config.hidden_size, config.n_mels_tgt, config.hidden_size
            )

    def init_vq_layers(self, suffix, config):
        if config.get("vq_type", None) == "CVQ":
            vq = ClusteredVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                distance=config.get("vq_distance", "cos"),
            )
        elif config.get("vq_type", None) == "FSQ":
            vq = FiniteScalarQuantizer(codebook_size=config.vq_codebook_size)
        elif config.get("vq_type", None) == "LFQ":
            vq = LookupFreeQuantizer(codebook_size=config.vq_codebook_size)
        elif config.get("vq_type", None) == "EMAEntropy":
            vq = EMAVectorQuantizerEntropy(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        else:
            vq = EMAVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        setattr(self, f"vq_{suffix}", vq)
        if config.get("vq_proj_norm", None) == "bn":
            vq_proj_in = nn.Sequential(
                Transpose(),
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity()
            )
        else:
            vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
        setattr(self, f"vq_proj_in_{suffix}", vq_proj_in)
        setattr(self, f"vq_proj_out_{suffix}", vq_proj_out)

        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer(f"cnt_{suffix}", torch.FloatTensor([0]))

    def forward_vq(self, hidden_states, suffix):
        org_len = hidden_states.shape[1]
        hidden_states = getattr(self, f"vq_proj_in_{suffix}")(hidden_states)
        cnt = getattr(self, f"cnt_{suffix}")
        if self.config.get("vq_proj_noise", 0) > 0:
            noise_scale = (self.config.vq_proj_noise - cnt).clamp(
                0
            ) / self.config.vq_proj_noise
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            cnt.add_(1)
        vq = getattr(self, f"vq_{suffix}")
        if self.config.get("vq_type", None) == "FSQ":
            vq_embs, vq_ids = vq(hidden_states)
            vq_loss = None
        elif self.config.get("vq_type", None) == "EMAEntropy":
            vq_embs, vq_ids, vq_loss = vq(
                hidden_states, e_scale=1.0 if cnt < 30_000 else 0.0
            )
        else:
            vq_embs, vq_ids, vq_loss = vq(hidden_states)
        hidden_states = getattr(self, f"vq_proj_out_{suffix}")(vq_embs)[:, :org_len]
        return hidden_states, vq_ids, vq_loss

    def forward_decoder(self, hidden_states, decoder, fea_ref=None, ref_enc=None):
        if self.config.vocal_ref and fea_ref is not None:
            hidden_states = ref_enc(hidden_states, fea_ref) + hidden_states
        hidden_states = decoder(hidden_states)
        return hidden_states

    def forward_inst_branch(self, input_dict):
        feature = input_dict["inst"]
        if self.config.get("use_full_input", False):
            feature = input_dict["full"]
        nonpadding = (feature.abs().sum(-1) > 0).float()[..., None]
        T = nonpadding.shape[1]

        # encoder & VQ
        hidden_states_inst = self.audio_encoder_inst(feature, nonpadding)
        hidden_states_inst = self.encoder_layer(hidden_states_inst)
        hidden_states_inst, vq_ids_inst, vq_loss_inst = self.forward_vq(
            hidden_states_inst, "inst"
        )
        hidden_states_aftervq_inst = hidden_states_inst = self.forward_decoder(
            hidden_states_inst, self.decoder_inst
        )

        # mel and chroma out
        mel_out_inst = self.mel_head_inst(hidden_states_inst, nonpadding)[:, :T]
        chroma_out_inst = self.chroma_head_inst(hidden_states_inst, nonpadding)[:, :T]
        output_dict = {
            "mel_out_inst": mel_out_inst,
            "vq_ids_inst": vq_ids_inst,
            "vq_loss_inst": vq_loss_inst,
            "chroma_out_inst": chroma_out_inst,
            "hidden_states_aftervq_inst": hidden_states_aftervq_inst,
        }
        return output_dict

    def forward_vocal_branch(self, input_dict):
        feature_vocal_f = feature_vocal_ref_f = input_dict["mel_vocal_f"]
        nonpadding_f = (feature_vocal_f.abs().sum(-1) > 0).float()[..., None]
        feature_vocal_b = feature_vocal_ref_b = input_dict["mel_vocal_b"]
        nonpadding_b = (feature_vocal_b.abs().sum(-1) > 0).float()[..., None]
        T_f = nonpadding_f.shape[1]
        T_b = nonpadding_b.shape[1]
        if self.config.get("use_full_input", False):
            feature_vocal_f = input_dict["full"][:, :T_f]
            feature_vocal_b = input_dict["full"][:, T_f:]

        # encoder & VQ
        hidden_states_vocal_f = self.audio_encoder_vocal(feature_vocal_f, nonpadding_f)
        hidden_states_vocal_f = self.encoder_layer(hidden_states_vocal_f, nonpadding_f)
        hidden_states_vocal_f, vq_ids_vocal_f, vq_loss_vocal_f = self.forward_vq(
            hidden_states_vocal_f, "vocal"
        )

        hidden_states_vocal_b = self.audio_encoder_vocal(feature_vocal_b, nonpadding_b)
        hidden_states_vocal_b = self.encoder_layer(hidden_states_vocal_b, nonpadding_b)
        hidden_states_vocal_b, vq_ids_vocal_b, vq_loss_vocal_b = self.forward_vq(
            hidden_states_vocal_b, "vocal"
        )

        # reference-based decoder
        if "mel_vocal_ref_f" in input_dict:
            feature_vocal_ref_f = input_dict["mel_vocal_ref_f"]
            feature_vocal_ref_b = input_dict["mel_vocal_ref_b"]

        hidden_states_vocal_f = self.forward_decoder(
            hidden_states_vocal_f,
            self.decoder_vocal,
            feature_vocal_ref_b,
            self.timbre_enc_vocal,
        )
        hidden_states_vocal_b = self.forward_decoder(
            hidden_states_vocal_b,
            self.decoder_vocal,
            feature_vocal_ref_f,
            self.timbre_enc_vocal,
        )
        hidden_states_vocal = torch.cat(
            [hidden_states_vocal_f, hidden_states_vocal_b], 1
        )

        # mel out
        mel_out_vocal_f = self.mel_head_vocal(hidden_states_vocal_f, nonpadding_f)[
            :, :T_f
        ]
        mel_out_vocal_b = self.mel_head_vocal(hidden_states_vocal_b, nonpadding_b)[
            :, :T_b
        ]
        mel_out_vocal = torch.cat([mel_out_vocal_f, mel_out_vocal_b], 1)

        # f0 out
        f0_vuv_out_vocal_f = self.f0_vuv_head_vocal(
            hidden_states_vocal_f, nonpadding_f
        )[:, :T_f]
        f0_vuv_out_vocal_b = self.f0_vuv_head_vocal(
            hidden_states_vocal_b, nonpadding_b
        )[:, :T_b]
        f0_vuv_out_vocal = torch.cat([f0_vuv_out_vocal_f, f0_vuv_out_vocal_b], 1)

        # for LAS loss
        text_ids = input_dict["text_ids"]
        text_ids_pad = F.pad(text_ids, [1, -1], value=1)
        text_out = self.asr_aux_decoder_vocal(
            hidden_states_vocal, None, text_ids_pad, (text_ids_pad > 0).sum(-1)
        )[0]
        if vq_loss_vocal_b is not None:
            vq_loss_vocal = vq_loss_vocal_b + vq_loss_vocal_f
        else:
            vq_loss_vocal = None
        vq_ids_vocal = torch.cat([vq_ids_vocal_f, vq_ids_vocal_b], 1)
        output_dict = {
            "mel_out_vocal": mel_out_vocal,
            "vq_ids_vocal": vq_ids_vocal,
            "vq_loss_vocal": vq_loss_vocal,
            "f0_out_vocal": f0_vuv_out_vocal[:, :, 0:1],
            "vuv_out_vocal": f0_vuv_out_vocal[:, :, 1:],
            "text_out_vocal": text_out,
            "hidden_states_aftervq_vocal": (
                hidden_states_vocal_f,
                hidden_states_vocal_b,
            ),
            "feature_vocal_ref": (feature_vocal_ref_f, feature_vocal_ref_b),
        }
        return output_dict

    def forward_full_branch(
        self, input_dict, h_aftervq_vocal, h_aftervq_inst, h_vocal_ref
    ):
        feature_full = input_dict["full"]
        nonpadding = (feature_full.abs().sum(-1) > 0).float()[..., None]
        T = nonpadding.shape[1]
        feature_vocal_f = input_dict["mel_vocal_f"]
        nonpadding_f = (feature_vocal_f.abs().sum(-1) > 0).float()[..., None]
        feature_vocal_b = input_dict["mel_vocal_b"]
        nonpadding_b = (feature_vocal_b.abs().sum(-1) > 0).float()[..., None]
        T_f = nonpadding_f.shape[1]
        T_b = nonpadding_b.shape[1]

        # combine afterVQ states
        hidden_states_vocal_f, hidden_states_vocal_b = h_aftervq_vocal
        T_f_hs = hidden_states_vocal_f.shape[1]
        if self.config.get("detach_vocal_to_full", False):
            hidden_states_vocal_f = hidden_states_vocal_f.detach()
            hidden_states_vocal_b = hidden_states_vocal_b.detach()
        hidden_states_full_f = torch.cat(
            [hidden_states_vocal_f, h_aftervq_inst[:, :T_f_hs]], -1
        )
        hidden_states_full_b = torch.cat(
            [hidden_states_vocal_b, h_aftervq_inst[:, T_f_hs:]], -1
        )
        feature_vocal_ref_f, feature_vocal_ref_b = h_vocal_ref
        hidden_states_full_f = self.decoder_in_proj(
            hidden_states_full_f.transpose(1, 2)
        ).transpose(1, 2)
        hidden_states_full_b = self.decoder_in_proj(
            hidden_states_full_b.transpose(1, 2)
        ).transpose(1, 2)
        hidden_states_full_f = self.forward_decoder(
            hidden_states_full_f,
            self.decoder_full,
            feature_vocal_ref_b,
            self.timbre_enc_full,
        )
        hidden_states_full_b = self.forward_decoder(
            hidden_states_full_b,
            self.decoder_full,
            feature_vocal_ref_f,
            self.timbre_enc_full,
        )
        hidden_states_full = torch.cat([hidden_states_full_f, hidden_states_full_b], 1)

        # mel out
        mel_out_full_f = self.mel_head_full(hidden_states_full_f, nonpadding_f)[:, :T_f]
        mel_out_full_b = self.mel_head_full(hidden_states_full_b, nonpadding_b)[:, :T_b]
        mel_out_full = torch.cat([mel_out_full_f, mel_out_full_b], 1)

        # f0 out
        f0_vuv_out_full_f = self.f0_vuv_head_full(hidden_states_full_f, nonpadding_f)[
            :, :T_f
        ]
        f0_vuv_out_full_b = self.f0_vuv_head_full(hidden_states_full_b, nonpadding_b)[
            :, :T_b
        ]
        f0_vuv_out_full = torch.cat([f0_vuv_out_full_f, f0_vuv_out_full_b], 1)

        # for LAS loss
        text_ids = input_dict["text_ids"]
        text_ids_pad = F.pad(text_ids, [1, -1], value=1)
        text_out = self.asr_aux_decoder_full(
            hidden_states_full, None, text_ids_pad, (text_ids_pad > 0).sum(-1)
        )[0]
        chroma_out_full = self.chroma_head_full(hidden_states_full, nonpadding)[:, :T]
        output_dict = {
            "mel_out_full": mel_out_full,
            "f0_out_full": f0_vuv_out_full[:, :, 0:1],
            "vuv_out_full": f0_vuv_out_full[:, :, 1:],
            "text_out_full": text_out,
            "chroma_out_full": chroma_out_full,
        }
        return output_dict

    def forward(self, input_dict, branches=None):
        if branches is None:
            branches = ["vocal", "inst", "full"]
        output_dict = {}
        if "vocal" in branches:
            output_dict_vocal = self.forward_vocal_branch(input_dict)
            output_dict.update(output_dict_vocal)
        if "inst" in branches:
            output_dict_inst = self.forward_inst_branch(input_dict)
            output_dict.update(output_dict_inst)
        if "full" in branches:
            output_dict_full = self.forward_full_branch(
                input_dict,
                output_dict_vocal.pop("hidden_states_aftervq_vocal"),
                output_dict_inst.pop("hidden_states_aftervq_inst"),
                output_dict_vocal.pop("feature_vocal_ref"),
            )
            output_dict.update(output_dict_full)
        output_dict["flops"] = 0
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav, wav_type="vocal"):
        wav = self._prepare_wav(wav)
        mel = torch_wav2spec(wav)
        nonpadding = (mel.abs().sum(-1) > 0).float()[..., None]
        if wav_type == "vocal":
            hidden_states = self.audio_encoder_vocal(mel, nonpadding)
        else:
            hidden_states = self.audio_encoder_inst(mel, nonpadding)
            wav_type = "inst"
        hidden_states = self.encoder_layer(hidden_states, nonpadding)
        hidden_states, vq_ids, vq_loss = self.forward_vq(hidden_states, wav_type)
        return vq_ids

    def token2mel(self, token, wav_ref=None, token_type="vocal"):
        if token_type == "full":
            if wav_ref is not None:
                wav_ref = self._prepare_wav(wav_ref)
                mel_ref = torch_wav2spec(wav_ref)
            else:
                mel_ref = None
            token_vocal = token[:, ::2]
            token_inst = token[:, 1::2]
            if isinstance(self.vq_vocal, FiniteScalarQuantizer):
                vq_vocal = self.vq_vocal.indices_to_codes(token_vocal)
            else:
                vq_vocal = self.vq_vocal.embedding(token_vocal)
            h_vocal = self.vq_proj_out_vocal(vq_vocal)
            h_vocal = self.forward_decoder(
                h_vocal, self.decoder_vocal, mel_ref, self.timbre_enc_vocal
            )
            if isinstance(self.vq_inst, FiniteScalarQuantizer):
                vq_inst = self.vq_inst.indices_to_codes(token_inst)
            else:
                vq_inst = self.vq_inst.embedding(token_inst)
            h_inst = self.vq_proj_out_inst(vq_inst)
            h_inst = self.forward_decoder(h_inst, self.decoder_inst)
            h = torch.cat([h_vocal, h_inst], -1)
            h = self.decoder_in_proj(h.transpose(1, 2)).transpose(1, 2)
            h = self.forward_decoder(
                h, self.decoder_full, mel_ref, self.timbre_enc_full
            )
            mel_out = self.mel_head_full(h)
            return mel_out
        elif token_type == "vocal":
            if wav_ref is not None:
                wav_ref = self._prepare_wav(wav_ref)
                mel_ref = torch_wav2spec(wav_ref)
            else:
                mel_ref = None
            vq = self.vq_vocal
            vq_out = self.vq_proj_out_vocal
            mel_head = self.mel_head_vocal
        else:
            assert token_type == "inst"
            vq = self.vq_inst
            vq_out = self.vq_proj_out_inst
            mel_head = self.mel_head_inst
        if isinstance(vq, FiniteScalarQuantizer):
            vq_embs = vq.indices_to_codes(token)
        else:
            vq_embs = vq.embedding(token)
        hidden_states = vq_out(vq_embs)
        if token_type == "vocal":
            hidden_states = self.forward_decoder(
                hidden_states, self.decoder_vocal, mel_ref, self.timbre_enc_vocal
            )
        else:
            hidden_states = self.forward_decoder(hidden_states, self.decoder_inst)
        mel_out = mel_head(hidden_states)
        return mel_out

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _prepare_wav(self, wav):
        """Check audio dimensions and pad."""
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        return self.pad_audio(wav.float())

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def pad_audio(self, x):
        rate = int(self.config.sample_rate / self.config.frame_rate) // 4 * self.ds
        if x.size(-1) % rate > 0:
            return F.pad(x, (0, rate - (x.size(-1) % rate)), "constant", 0)
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, audio_dict):
        """
        Arguments()
            audio_dict:
                `audio`: full mix should be routed to normal mel spectrogram, chroma and pitch.
                `audio_vocal`: routed to a vocal mel spectrogram
                `audio_inst`: routed to a instruemtnal mel spectrogram

        Returns:
            input_dict:
                "mel": mel spectrogram of audio full mix
                "mel_vocal" : mel spectrogram of audio vocals
                "mel_inst" : mel spectrogram of audio instrumental
                "chroma" : chroma spectrogram of audio full mix

        @hanoihantrakul 11-25-2023
        The logic of this code should be read in conjunction with `lit_module.Stage2MSS().prepare_feature()`
        """

        def _interfere_audio_handler(audio):
            """Not used in UMM training."""
            audio_interfered = self.interfere_audio(audio)
            mel_interfered = self.audio_transform(audio_interfered, normalize=normalize)
            return mel_interfered

        def _add_chroma_handler(audio):
            chroma = self.chroma_transform(audio)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            return chroma

        def _add_pitch_handler(audio):
            """Not used in UMM training."""
            f0 = self.rmvpe.batch_infer(
                audio, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            return f0, vuv

        normalize = self.config.feature_cmvn is not None
        input_dict = {}
        if self.config.get("interfere_audio", None):
            # "interfere_audio" is a historical flag and should be assumed to be False by default.
            input_dict.update(
                mel_interfered=_interfere_audio_handler(audio_dict["audio"])
            )
        if self.config.add_chroma:
            input_dict.update(chroma=_add_chroma_handler(audio_dict["audio_inst"]))
        if self.config.get("add_pitch", False):
            # "add_pitch" is from TTS team and should be assumed to be False by default.
            f0, vuv = _add_pitch_handler(audio_dict["audio_vocal"])
            input_dict.update(f0=f0, vuv=vuv)

        return input_dict
