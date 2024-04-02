import torch
from torch import nn
from torch.nn import functional as F

from recipes.umm.models.dualumm_encoders import (
    ConvStacksWithDownUpSampling,
    MultiRefTimbreEncoder,
)
from recipes.umm.models.dualumm_vector_quantizers import (
    get_embeddings_from_vector_quantizer,
    get_noise_scale,
    get_vector_quantizer,
    get_vector_quantizer_projection_layers,
)
from recipes.umm.models.rmvpe import RMVPE
from recipes.umm.models.umm_mkii import get_vuv
from recipes.umm.models.wenet.transformer.decoder import TransformerDecoder
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.utils.mel_utils import torch_wav2spec


class DualUMMv2(nn.Module):
    """
    @renyi @hanoihantrakul 27 March 2024
    DualUMM is a 3rd generation tokenizer that has two separate branches:
    one for vocal and one for instrumental. These branches output their
    own respective codebooks, meaning there is a separate vocal codebook
    and separate instrumental codebook. This is different from 1st Gen
    ConformerUMM and 2nd Gen ConvUMM where there is a single codebook
    for both vocal and instrumental audio.

    DualUMMv2 requires the input to already be separated using an offline
    MSS model into vocal and instrumental parts. (DualUMMv1 operates directly
    on the full mix, but we found the performance to not be as good as the one
    where an MSS model pre-separates the input)

    From a high level perspective, DualUMMv2 introduces losses specific to
    the vocal branch (e.g. ASR-based losses) and instrumental branch (e.g.
    chroma spectrogram). In addition, there is a third branch which combines
    the vector quantized values from both branches and outputs the full mix.
    A range of additional losses (e.g. adversarial losses, Structure Similarity
    Index SSIM loss) are imposed on this full mix. This forces more
    information to be compressed into a single token.

    Since there is a vocal token and instrumental token, remember to add these
    two frame rates together to get the effective frame rate. e.g. if the
    vocal codes are 10Hz and the instrumental codes are 10Hz, then the overall
    frame rate after concatenation or interleaving of this DualUMM model is 20Hz.

    As of 27 March 2024, we have verified that this model outperforms Gen1 and Gen2
    tokenizers for tasks involving vocals and instrumentals (e.g. Lyrics2Song)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ds = config.get(
            "downsampling", 4
        )  # The mel features are at frame rate 100. So downsampling=4 means each branch is 25Hz.
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
        """Initialize the instrumental branch audio_encoder, decoders and reconstruction heads."""
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
        """Initialize the vocal branch. It has different encoders, decoders and recon heads to the instrumental branch."""
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
        """Initialize the full mix branch, which only has a decoder and recon heads."""
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
        """Initliaze the VQ layer."""
        # Configure the specific type of VQ
        vq_type = config.get("vq_type", None)
        vq = get_vector_quantizer(vq_type, config)
        setattr(self, f"vq_{suffix}", vq)

        # Configure the type of projection layer going into and out of VQ layer
        vq_proj_norm_type = config.get("vq_proj_norm", None)
        vq_proj_in, vq_proj_out = get_vector_quantizer_projection_layers(
            vq_proj_norm_type, config
        )
        setattr(self, f"vq_proj_in_{suffix}", vq_proj_in)
        setattr(self, f"vq_proj_out_{suffix}", vq_proj_out)

        # Configure the noise injected into the VQ projection layer
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer(f"cnt_{suffix}", torch.FloatTensor([0]))

    def forward_vq(self, hidden_states, suffix):
        """Forward pass through the VQ layer."""
        org_len = hidden_states.shape[1]
        # Apply projection into VQ layer
        hidden_states = getattr(self, f"vq_proj_in_{suffix}")(hidden_states)
        cnt = getattr(self, f"cnt_{suffix}")
        # Apply projection noise
        vq_proj_noise = self.config.get("vq_proj_noise", 0)
        if vq_proj_noise > 0:
            noise_scale = get_noise_scale(vq_proj_noise, cnt)
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            cnt.add_(1)
        # Apply VQ
        # TODO(@hanoihantrakul): wrap different VQ behaviors with a consistent signature
        vq = getattr(self, f"vq_{suffix}")
        vq_type = self.config.get("vq_type", None)
        if vq_type == "FSQ":
            vq_embs, vq_ids = vq(hidden_states)
            vq_loss = None
        elif vq_type == "EMAEntropy":
            vq_embs, vq_ids, vq_loss = vq(
                hidden_states, e_scale=1.0 if cnt < 30_000 else 0.0
            )
        else:
            vq_embs, vq_ids, vq_loss = vq(hidden_states)
        # Apply projection out of VQ layer
        hidden_states = getattr(self, f"vq_proj_out_{suffix}")(vq_embs)[:, :org_len]
        return hidden_states, vq_ids, vq_loss

    def forward_decoder(self, hidden_states, decoder, fea_ref=None, ref_enc=None):
        """Helper function for forward pass through the decoder layer."""
        if fea_ref is not None:
            # Use the reference feature and encoder. Normally off by defaul
            hidden_states = ref_enc(hidden_states, fea_ref) + hidden_states
        hidden_states = decoder(hidden_states)
        return hidden_states

    def forward_inst_branch(self, input_dict):
        """Forward pass for instrumental branch."""
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
        """Forward pass for vocal branch.

        - Implementation notes @renyi @hanoihantrakul 27 Mar 2024
        The vocal is split into two parts, a "front" and "back" part.
        In the code these are denoted by `_f` and `_b`. A parameter in the
        corresponding lit_module called `T_split` controls where this happens.

        In practise, this feature is not used by default. You can think of the
        vocal audio as passing through this branch "as a single file".

        It was written this way to support experimentation  where timbre and content disentanglement
        can be enforced by getting the model to predict features of the front
        using features of the back portion (e.g. use the tone/timbre of the back, but not
        the content).
        """
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

        use_vocal_ref = self.config.get(
            "vocal_ref", False
        )  # For nearly all cases this is False.
        if use_vocal_ref:
            # False by default
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
        else:
            # Normal scenario
            hidden_states_vocal_f = self.forward_decoder(
                hidden_states_vocal_f, self.decoder_vocal
            )
            hidden_states_vocal_b = self.forward_decoder(
                hidden_states_vocal_b, self.decoder_vocal
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
        """Forward pass for full mix branch.

        - Implementation notes @renyi @hanoihantrakul 27 Mar 2024
        Like the vocal branch, the code is structed to handle the "front"
        and "back" of the vocal input. This is to support experimentation
        in the  research model. In practise, the vocal audio passes through
        as though it was "one piece/
        """
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
            # When gradients are detached from the vocal,
            # we hope less timbre information is encoded in vocal branch
            # and can be controlled elsewhere
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

        # For nearly all cases this is False.
        use_vocal_ref = self.config.get("vocal_ref", False)
        if use_vocal_ref:
            # False by default
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
        else:
            # Normal scenario
            hidden_states_full_f = self.forward_decoder(
                hidden_states_full_f, self.decoder_full
            )
            hidden_states_full_b = self.forward_decoder(
                hidden_states_full_b, self.decoder_full
            )
        # Concatenate hidden states from front and back
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
        elif wav_type == "inst":
            hidden_states = self.audio_encoder_inst(mel, nonpadding)
        else:
            raise ValueError(
                f"Please choose either 'vocal' or 'inst' as wav_type instead of {wav_type}"
            )
        hidden_states = self.encoder_layer(hidden_states, nonpadding)
        hidden_states, vq_ids, vq_loss = self.forward_vq(hidden_states, wav_type)
        return vq_ids

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def token2mel(self, token, wav_ref=None, token_type="vocal"):
        """Convert tokens to a mel spectrogram."""

        def _vocal_tokens_to_hidden_states_helper(token, wav_ref):
            """Convert vocal tokens to vocal hidden states after VQ depending on presence of a vocal red"""
            if wav_ref is not None:
                # Use the `wav_ref` as a reference timbre. Normally false.
                wav_ref = self._prepare_wav(wav_ref)
                mel_ref = torch_wav2spec(wav_ref)
                return _vocal_tokens_to_hidden_states_after_vq_with_ref(token, mel_ref)
            else:
                return _vocal_tokens_to_hidden_states_after_vq_no_ref(token)

        def _vocal_tokens_to_hidden_states_after_vq_no_ref(token):
            """Convert vocal tokens to vocal hidden states after VQ with no reference vocal."""
            vq_embs = get_embeddings_from_vector_quantizer(token, self.vq_vocal)
            hidden_states = self.vq_proj_out_vocal(vq_embs)
            hidden_states = self.forward_decoder(hidden_states, self.decoder_vocal)
            return hidden_states

        def _vocal_tokens_to_hidden_states_after_vq_with_ref(token, mel_ref):
            """Convert vocal tokens to vocal hidden states after VQ with a reference vocal."""
            vq_embs = get_embeddings_from_vector_quantizer(token, self.vq_vocal)
            hidden_states = self.vq_proj_out_vocal(vq_embs)
            hidden_states = self.forward_decoder(
                hidden_states, self.decoder_vocal, mel_ref, self.timbre_enc_vocal
            )
            return hidden_states

        def _instrumental_tokens_to_hidden_states_after_vq(token):
            """Convert instrumental tokens to instrumental hidden states after VQ."""
            vq_embs = get_embeddings_from_vector_quantizer(token, self.vq_inst)
            hidden_states = self.vq_proj_out_inst(vq_embs)
            hidden_states = self.forward_decoder(hidden_states, self.decoder_inst)
            return hidden_states

        if token_type == "vocal":
            h_vocal = _vocal_tokens_to_hidden_states_helper(token, wav_ref)
            return self.mel_head_vocal(h_vocal)
        elif token_type == "inst":
            h_inst = _instrumental_tokens_to_hidden_states_after_vq(token)
            return self.mel_head_inst(h_inst)
        elif token_type == "full":
            # Handle vocal tokens
            token_vocal = token[:, ::2]
            h_vocal = _vocal_tokens_to_hidden_states_helper(token_vocal, wav_ref)
            # Handle instrumental tokens
            token_inst = token[:, 1::2]
            h_inst = _instrumental_tokens_to_hidden_states_after_vq(token_inst)
            # Concatenate tokens
            h_full = torch.cat([h_vocal, h_inst], -1)
            # Decode into a full mix
            h_full = self.decoder_in_proj(h_full.transpose(1, 2)).transpose(1, 2)
            if wav_ref is not None:
                # Use the `wav_ref` as a reference timbre. Normally false.
                wav_ref = self._prepare_wav(wav_ref)
                mel_ref = torch_wav2spec(wav_ref)
                h_full = self.forward_decoder(
                    h_full, self.decoder_full, mel_ref, self.timbre_enc_full
                )
            else:  # Normal Scenario
                h_full = self.forward_decoder(h_full, self.decoder_full)
            return self.mel_head_full(h_full)
        else:
            raise ValueError("token_type must be one of [vocal, inst, full]")

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

        @hanoihantrakul 2 April 2024
        Note that DualUMMv2 defines the mel spec transform in the lit_module
        with `process_tgt_mel()` not here in the model.preprocessing() method.
        It is different from previous ConformerUMM and ConvUMM pipeline.
        """

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

        input_dict = {}
        if self.config.add_chroma:
            input_dict.update(chroma=_add_chroma_handler(audio_dict["audio_inst"]))
        if self.config.get("add_pitch", False):
            # "add_pitch" is from TTS team and should be assumed to be False by default.
            f0, vuv = _add_pitch_handler(audio_dict["audio_vocal"])
            input_dict.update(f0=f0, vuv=vuv)

        return input_dict


class DualUMMv2Inst(DualUMMv2):
    """
    DualUMM adapted for Instrumental only branch.

    Implementation Notes
    @hanoihantrakul 28 March 2024
    You might wonder "isn't a single branch DualUMM just a regular UMM?"
    The main difference between this implementation and a traditional ConvUMM is:
    - the addition of GAN adversarial losses on the recon heads (main difference)
    - the location of the VQ layer (halfway between encoders in ConvUMM
    and after encoder in DualUMM)
    """

    def __init__(self, config):
        """
        @hanoihantrakul 29Mar2024 TODO: if this works, recommend refactoring a separate InstrumentalBranch() class that
        DualUMM can inherit for vocal and instrumental branches, and this instrumental-only
        model can call separately.
        """
        # Do not call parent __init__() method which would init the vocal branch. We just want to inherit the methods.
        self.config = config
        self.ds = config.get(
            "downsampling", 4
        )  # The mel features are at frame rate 100. So downsampling=4 means each branch is 25Hz.
        self.us = config.get("upsampling", 1)
        self.conv_hidden_size = config.get("conv_hidden_size", 256)
        self.encoder_layer = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        self.init_inst_branch(config)
        self.chroma_transform = ChromaSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_chroma=config.n_chroma,
            normalized=False,
        )

    def forward(self, input_dict):
        """Forward pass only needs to call instrumental branch."""
        output_dict = self.forward_inst_branch(input_dict)
        output_dict["flops"] = 0
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def token2mel(self, token):
        """Convert tokens to a mel spectrogram."""
        return super.token2mel(token, token_type="inst")

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        return super.wav2token(wav, wav_type="inst")

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        """
        Implementation Notes
        2 April 2024 @hanoihantrakul
        This method needs to be different from DualUMMv2.preprocessing() because
        it only trains on instrumental data. The logic is closer to
        the older ConformerUMM and ConvUMM training pipeline. I copy-pasted
        from `umm_mkii.Stage2.preprocessing()` to accomplish this.

        However, note that DualUMMv2 defines the mel spec transform in the lit_module
        with `process_tgt_mel()` not here in the model.preprocessing() method.
        It is different from previous ConformerUMM and ConvUMM pipeline.
        """
        # normalize = self.config.feature_cmvn is not None
        # mel = self.audio_transform(x, normalize=normalize)
        # input_dict = {"mel": mel}
        input_dict = {}
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        return input_dict
