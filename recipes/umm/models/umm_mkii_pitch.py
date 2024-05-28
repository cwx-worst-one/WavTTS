import torch
from torch.nn import functional as F

from recipes.umm.models.dualumm_encoders import ConvStacksWithDownUpSampling
from recipes.umm.models.dualumm_vector_quantizers import get_vq_codebook_distances
from recipes.umm.models.umm_mkii import Conv1dUpsampling, Stage3, Stage3MSSDec
from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.models.voc_modules.pitch_predictor.inference import (
    PerceptualPitchPredictor,
)
from recipes.umm.utils.mel_utils import torch_wav2spec

"""
@hanoihantrakul 22MAY2024 Implementation Notes for `umm_mkii_pitch.py`
We want to test the effect of adding supervised and perceptual pitch losses to
a tokenizer trained for vocal music. At the time of writing, the only experiments
I ever did which incorporated this were the old ConformerUMM tokenizers before
we migrated to ConvUMM and ConvUMM-GAN architectures. These were done in JAN2024.
    
In revisiting this question I note that the inheritance structure in `umm_mkii.py`
is very confusing. Thus I have created this new file which is similar to `umm_mkii.Stage3Conv1D_v2`
but will correctly handle the additional cases of adding a supervised pitch loss (SPL)
and perceptual pitch loss (PPL).
"""


class Stage3Conv1D_v2Pitch(Stage3):
    """
    @hanoihantrakul 22MAY2024
    At the time of writing, the L2S Vocal pipeline uses tokenizers based on `Stage3Conv1D_v2`
    and hadn't yet migrated to `ConvUMM-GAN` implementation. At the time of writing, the SSTK
    V9 Product Model had already made this migration. Thus to properly test the effects of
    adding SPL and PPL, I have based this implementation off `umm_mkii.Stage3Conv1D_v2`.

    To avoid a confusing inheritance structure, I have based this off `umm_mkii.Stage3` (Conformer based)
    and added all the neccesary logic to make this into a Conv1D model with additional
    supervised and perceptual pitch inits, heads and losses.
    """

    def __init__(self, config):
        """
        Replace self.audio_encoder with DualUMM-style ConvEncoder.

        @hanoihantrakul 29Feb2024:
        - Note that the old self.audio_encoder used
        umm_mkii.Conv2dSubsampling() with 2 sets of stride 2 i.e 4x downsampling.
        Here we use dualumm.ConvStacksWithDownUpSampling() with identical
        4x downsampling.

        - Another difference is dualumm.ConvStacksWithDownUpSampling() targets
        a mel 160 spectrogram, whereas umm_mkii.Conv2dSubsampling() targets a
        mel 128 spectrogram. The reason is only because I want to use the same
        functions developed in DualUMM.
        """
        super().__init__(config)
        # Remove Conv1D based initialization from default Stage3 model
        del self.audio_encoder
        self.audio_encoder = ConvStacksWithDownUpSampling(
            config.hidden_size,
            config.n_mels_tgt,
            config.hidden_size,
            downsampling=config.downsampling,
            upsampling=config.upsampling,
        )

        # Remove mel-128 from default Stage3 audio transform and replace with mel-160
        del self.audio_transform
        self.audio_transform = lambda x: torch_wav2spec(
            x, num_mels=config.n_mels_tgt, sample_rate=config.sample_rate
        )

        # Mel Head to Conv1D. Replace all Conv2DUpsampling from default Stage3 model with Conv1DUpsampling
        del self.mel_head
        self.mel_head = Conv1dUpsampling(
            config.hidden_size,
            config.n_mels_tgt,
            act_fn=torch.nn.ReLU
            if config.get("act_fn", "relu") == "relu"
            else torch.nn.GELU,
        )

        # Chroma Head to Conv1D. Replace all Conv2DUpsampling from default Stage3 model with Conv1DUpsampling
        if config.add_chroma:
            del self.chroma_head
            self.chroma_head = Conv1dUpsampling(
                config.hidden_size,
                config.n_chroma,
                act_fn=torch.nn.ReLU
                if config.get("act_fn", "relu") == "relu"
                else torch.nn.GELU,
            )

        # Supervised Pitch Case with Conv1DUpsampling
        if config.get("add_supervised_pitch", False):
            """Use in house pitch predictor for supervised loss in BigMusic pipeline. This is different from TTS, which uses RVMPE as supervised loss."""
            # the pl_module handles loading the pretrained state_dict of perceptual pitch predictor
            self.pitch_predictor = PerceptualPitchPredictor()
            # This head reconstructs the f0_hz and vuv signals from a mel-160 spectrogram
            self.f0_vuv_head = Conv1dUpsampling(
                config.hidden_size, 2
            )  # Replace all Conv2DUpsampling with Conv1DUpsampling

        # Perceptual Pitch Case
        if self.config.get("add_perceptual_pitch", False):
            """Replace original mel spec reconstruction head (n_mel=128) with perceptual loss reconstruction head (n_mel=160)."""
            # the pl_module handles loading the pretrained state_dict of perceptual pitch predictor
            self.pitch_predictor = PerceptualPitchPredictor()
            # @hanoihantrakul 27MAY2024 There is no special reconstruction head for perceptual loss.
            # We pass the mel-160 into the pitch predictor and then use an L1 loss on the hidden states directly (see pl_module)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        """
        Preprocessing on audio. Add mel, chroma and pitch related features.
        """
        mel = self.audio_transform(x)  # mel-160 features
        input_dict = {"mel": mel}
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        if self.config.get("add_supervised_pitch", False):
            """
            @hanoihantrakul 27MAY2024 Original Supervised Pitch Loss was only used by TTS, not BigMusic pipeline.
            You can see TTS implementation here:
            `
            f0 = self.rmvpe.batch_infer(
                x, self.config.sample_rate, thred=0.03, use_viterbi=False
            )
            f0 = f0[:, :-1]
            vuv = get_vuv(f0)
            f0 = f0_normalize(f0)
            input_dict.update(f0=f0, vuv=vuv)
            `
            For BigMusic, we do things differently. First, we use our own in-house trained pitch predictor and not RVMPE (open source)
            - The function f0_normalize(f0) centers the signal with mean=0 and std=1 since TTS only cares about relative deviations in pitch.
            We do not want this in vocal music.
            - The function get_vuv() outputs a binary state. I apply the same logic to our in-house pitch predictor.
            """
            pd_output = self.pitch_predictor.forward(
                input_dict["mel"]
            )  # use same mel-160 as the spectrogram loss
            f0 = pd_output[
                :, :, 0
            ]  # this output by default is in a log scale. See `recipes/umm/modules/pitch_predictor_task.compute_ground_truth_pitch()`
            vuv = pd_output[:, :, 1]
            vuv = pitch_utils.post_process_vuv_logits_to_binary_state(
                vuv
            )  # vuv is now a binary state like the output of RVMPE
            input_dict.update(
                f0=f0.unsqueeze(2), vuv=vuv.unsqueeze(2)
            )  # add back dimension [batch_size, time_steps, 1] for loss computation later on
        if self.config.get("add_perceptual_pitch", False):
            """
            @hanoihantrakul 27MAY2024
            I add this to make it explicit that no additional preprocessing
            is required for perceptual pitch loss.
            """
            pass
        return input_dict

    def forward(self, input_dict):
        """
        @hanoihantrakul 3Mar2024
        Override Default Stage3 forward:
        - removing positional embedding and adding length checks

        @hanoihantrakul 5MAY2024
        - add supervised and perceptual pitch loss branches
        """
        feature = input_dict["mel"]  # this is mel-160
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)

        # TODO: @hanoihantrakul 27MAY2024 this can be simplified to be like `convumm_gan.forward_vq()`
        for i, layer in enumerate(self.encoder_layers):
            # at specific layer idx, vector quantize hidden states before applying the layer
            if i == self.config.vq_layer_idx:
                hidden_states = self.vq_proj_in(hidden_states)
                if self.config.get("vq_proj_noise", 0) > 0:
                    noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(
                        0
                    ) / self.config.vq_proj_noise
                    hidden_states = (
                        hidden_states + torch.randn_like(hidden_states) * noise_scale
                    )
                    self.cnt.add_(1)
                if self.config.get("vq_type", None) == "FSQ":
                    vq_embs, vq_ids = self.vq(hidden_states)
                    vq_loss = None
                elif self.config.get("vq_type", None) == "EMAEntropy":
                    vq_embs, vq_ids, vq_loss = self.vq(
                        hidden_states, e_scale=1.0 if self.cnt < 30_000 else 0.0
                    )
                else:
                    # default
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                    # calculate codebook distances by accessing the VQ's internal matrix representing the actual codebook
                    codebook_distance_stats = get_vq_codebook_distances(
                        self.vq.embedding.weight.data
                    )
                hidden_states = self.vq_proj_out(vq_embs)
            # apply layer
            hidden_states = layer(hidden_states)
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)

        """
        @hanoihantrakul 3Mar2024
        `mel` (ground truth) can sometimes be 1 sample longer than `mel_out` (predicted)
        - Just for this one config only, correct for this 1 sample difference.
        - e.g. [6, 2917, 160] vs [6, 2916, 160]
        - This will not be a problem if a compeletely new class is defined without inheritance from Stage1 and Stage2
        """
        mel_trim_len = pitch_utils.compute_min_lengths(input_dict["mel"], mel_out, axis=1)
        input_dict["mel"] = input_dict["mel"][:, :mel_trim_len, :]
        mel_out = mel_out[:, :mel_trim_len, :]

        output_dict = {
            "mel_out": mel_out,
            "vq_ids": vq_ids,
            "vq_loss": vq_loss,
            "flops": flops * 3,  # extra 2x for backward.
        }
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.get("vq_proj_noise", False):
            output_dict.update(noise_scale=noise_scale)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_supervised_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            f0_out=f0_vuv_out[:, :, 0:1]
            vuv_out=f0_vuv_out[:, :, 1:]
            # sometimes f0_gt, vuv_gt is 1 timestep longer than f0_out, vuv_out
            f0_vuv_trim_len = pitch_utils.compute_min_lengths(f0_out, input_dict['f0'], axis=1)
            input_dict["f0"] = input_dict["f0"][:, :f0_vuv_trim_len, :]
            f0_out = f0_out[:, :f0_vuv_trim_len, :]
            input_dict["vuv"] = input_dict["vuv"][:, :f0_vuv_trim_len, :]
            vuv_out = vuv_out[:, :f0_vuv_trim_len, :]
            output_dict.update(
                f0_out=f0_out
            )  # [batch_size, time_steps, 1]
            output_dict.update(
                vuv_out=vuv_out
            )  # [batch_size, time_steps, 1]
        if self.config.get("add_perceptual_pitch", False):
            # get predicted hidden state from reconstructed mel-160 spectrogram
            _ = self.pitch_predictor.forward(mel_out)
            h_pred = self.pitch_predictor.get_hidden_state()  # TODO: flops calculation
            # get reference hidden state from original data
            _ = self.pitch_predictor.forward(input_dict["mel"])
            h_gt = self.pitch_predictor.get_hidden_state()  # TODO: flops calculation
            # sometimes h_gt is 1 timestep longer than h_pred
            trim_len = pitch_utils.compute_min_lengths(h_pred, h_gt, axis=1)
            h_pred, h_gt = h_pred[:, :trim_len, :], h_gt[:, :trim_len, :]
            output_dict.update(
                h_pred=h_pred, h_gt=h_gt
            )  # UMMLoss class will handle this later in the pl_module
        # add the codebook stats
        output_dict.update(codebook_distance_stats)
        return output_dict
