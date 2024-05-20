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
    get_vq_codebook_distances,
    get_vq_losses,
)
from recipes.umm.models.wenet.transformer.decoder import TransformerDecoder
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.utils.mel_utils import torch_wav2spec


class ConvUMMGAN(nn.Module):
    """
    @hanoihantrakul 14MAY2024
    ConvUMM-GAN model is a:
    - single codebook
    - conv-based tokenizer
    - with additional GAN and SSIM loss
    - Mel, Chroma loss (Instrumental-only)
    - LAS loss (When training on vocal music)
    """

    def __init__(self, config):
        """
        @hanoihantrakul 29Mar2024 TODO: if this works, recommend refactoring a separate InstrumentalBranch() class that
        DualUMM can inherit for vocal and instrumental branches, and this instrumental-only
        model can call separately.

        1 April 2024
        Initially I tried to inherit from DualUMMv2 directly but it got messy very quickly. I was essententially
        initializing and deleting a bunch of uneeded encoders and branches.
        I have copy pasted methods and will refactor this in the future.
        """
        super().__init__()
        self.config = config
        # The mel features are at frame rate 100. So downsampling=4 means each branch is 25Hz.
        self.ds = config.get("downsampling", 4)
        self.us = config.get("upsampling", 1)
        self.conv_hidden_size = config.get("conv_hidden_size", 256)
        self.init_main_branch(config)
        self.init_transforms(config)

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
    def preprocessing(self, x):
        """
        Implementation Notes
        @hanoihantrakul 14MAY2024
        Ideally, the mel_transform() should be invoked here to keep the preprocessing logic consistent.
        Right now it happens in `lit_module_convumm_gan.prepare_features()` because I adapted the code
        from DualUMM codebase and that was the convention there.
        """

        def _add_chroma_handler(audio):
            chroma = self.chroma_transform(audio)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            return chroma

        input_dict = {}
        if self.config.add_chroma:
            input_dict.update(chroma=_add_chroma_handler(x))

        return input_dict

    def init_main_branch(self, config):
        """Initialize audio_encoder, decoders and reconstruction heads."""
        ### These layers are required for both instrumental and vocal music
        self.init_vq_layers(config)
        # The `input_spectrogram_encoder` performs the downsampling (self.ds) of Mel Spec to desired frame rate
        self.input_spectrogram_encoder = ConvStacksWithDownUpSampling(
            config.hidden_size,
            config.n_mels_tgt,
            config.hidden_size,
            downsampling=self.ds,
            upsampling=self.us,
        )
        # Historically this `self.encoder` was shared between two branches of DualUMM.
        # Here, you could technically do self.encoder = nn.Identity() and the system
        # would still work. This separation of `encoder` enables you to do
        # additional processing that is distinct from the downsampling conv
        # in the `input_spectrogram_encoder` (e.g. replace this with a non-conv architecture)
        self.encoder = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        self.decoder = ConvStacksWithDownUpSampling(
            config.hidden_size, config.hidden_size, config.hidden_size
        )
        # Define the reconstruction heads
        self.mel_head = ConvStacksWithDownUpSampling(
            self.conv_hidden_size,
            config.hidden_size,
            config.n_mels_tgt,
            downsampling=self.us,
            upsampling=self.ds,
        )
        self.chroma_head = ConvStacksWithDownUpSampling(
            self.conv_hidden_size,
            config.hidden_size,
            config.n_chroma,
            downsampling=self.us,
            upsampling=self.ds,
        )
        # This is a special recon head that should only be engaged for vocal music
        if config.get("train_on_vocal_music", False):  # TODO: ADD THIS TO CONFIG
            self.asr_aux_vocal_head = TransformerDecoder(
                config.vocab_size, config.hidden_size, 4, config.hidden_size * 4, 4
            )

    def init_vq_layers(self, config):
        """Initliaze the VQ layer."""
        # Configure the specific type of VQ
        vq_type = config.get("vq_type", None)
        self.vq = get_vector_quantizer(vq_type, config)
        # Configure the type of projection layer going into and out of VQ layer
        vq_proj_norm_type = config.get("vq_proj_norm", None)
        self.vq_proj_in, self.vq_proj_out = get_vector_quantizer_projection_layers(
            vq_proj_norm_type, config
        )
        # Configure the noise injected into the VQ projection layer
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer(f"cnt", torch.FloatTensor([0]))

    def init_transforms(self, config):
        """Initliaze audio-related transforms."""
        self.chroma_transform = ChromaSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_chroma=config.n_chroma,
            normalized=False,
        )
        # 14MAY2024 @hanoihantrakul: Ideally, the mel_transform() should be defined here for increased legibility. I will do it in the next MR.
        # self.mel_transform = torch_wav2spec()

    def forward(self, input_dict):
        """Main forward function."""
        output_dict = self.forward_main_branch(input_dict)
        output_dict[
            "flops"
        ] = 0  # historical requirement from original tokenizer lit_module training loop
        return output_dict

    def forward_main_branch(self, input_dict):
        """Forward pass for main branch."""
        feature = input_dict["mel"]
        nonpadding = (feature.abs().sum(-1) > 0).float()[..., None]
        T = nonpadding.shape[1]

        # apply encoders
        hidden_states = self.input_spectrogram_encoder(feature, nonpadding)
        hidden_states = self.encoder(hidden_states)
        # apply VQ
        (
            hidden_states_aftervq,
            vq_ids,
            vq_loss,
            vq_codebook_distance_stats,
        ) = self.forward_vq(hidden_states)
        # apply decoder
        hidden_states_aftervq = self.decoder(hidden_states_aftervq)

        # mel and chroma out
        mel_out = self.mel_head(hidden_states_aftervq, nonpadding)[:, :T]
        chroma_out = self.chroma_head(hidden_states_aftervq, nonpadding)[:, :T]

        # for LAS loss when training on vocals
        if self.config.get("train_on_vocal_music", False):
            text_ids = input_dict["text_ids"]
            text_ids_pad = F.pad(text_ids, [1, -1], value=1)
            text_out = self.asr_aux_vocal_head(
                hidden_states_aftervq, None, text_ids_pad, (text_ids_pad > 0).sum(-1)
            )[0]

        # Populate the ouput dictionary
        output_dict = {
            "mel_out": mel_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "chroma_out": chroma_out,
            "hidden_states_aftervq": hidden_states_aftervq,  # TODO: CHANGE VAR NAME
        }
        output_dict.update(vq_codebook_distance_stats)  # add codebook distance stats
        # Add text output when training on vocal music
        if self.config.get("train_on_vocal_music", False):
            output_dict.update(text_out=text_out)
        return output_dict

    def forward_vq(self, hidden_states):
        """Forward pass through the VQ layer."""
        org_len = hidden_states.shape[1]
        # Apply projection into VQ layer
        hidden_states = self.vq_proj_in(hidden_states)
        # Apply projection noise and keep track of the counter
        cnt = getattr(self, f"cnt")
        vq_proj_noise = self.config.get("vq_proj_noise", 0)
        if vq_proj_noise > 0:
            noise_scale = get_noise_scale(vq_proj_noise, cnt)
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            cnt.add_(1)
        # Apply VQ
        vq_type = self.config.get("vq_type", None)
        # Inject logic for getting codebook distance
        codebook_distance_stats = get_vq_codebook_distances(
            self.vq.embedding.weight.data
        )
        vq_embs, vq_ids, vq_loss = get_vq_losses(self.vq, vq_type, hidden_states, cnt)
        # Apply projection out of VQ layer
        hidden_states = self.vq_proj_out(vq_embs)[:, :org_len]
        return hidden_states, vq_ids, vq_loss, codebook_distance_stats

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """Convert input audio waveform to tokens."""
        wav = self._prepare_wav(wav)
        mel = torch_wav2spec(
            wav
        )  # 14MAY2024 @hanoihantrakul: Ideally this should reuse `lit_module_convumm_gan.process_tgt_mel` but right now it leads to circular import.
        nonpadding = (mel.abs().sum(-1) > 0).float()[..., None]
        hidden_states = self.input_spectrogram_encoder(mel, nonpadding)
        hidden_states = self.encoder(hidden_states, nonpadding)
        _, vq_ids, _, _ = self.forward_vq(hidden_states)
        return vq_ids

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def token2mel(self, token):
        """Convert tokens to a mel spectrogram."""
        vq_embs = get_embeddings_from_vector_quantizer(token, self.vq)
        hidden_states = self.vq_proj_out(vq_embs)
        hidden_states = self.decoder(hidden_states)
        return self.mel_head(hidden_states)
