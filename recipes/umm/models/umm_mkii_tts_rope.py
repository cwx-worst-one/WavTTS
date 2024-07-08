import torch
from torch import Tensor, int32, nn
from torch.nn import functional as F

from recipes.umm.models.dualumm_vector_quantizers import get_vq_codebook_distances
from recipes.umm.models.umm_mkii import Conv2dUpsampling, Stage2
from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.models.voc_modules.pitch_predictor.inference import (
    PerceptualPitchPredictor,
)
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.utils.mel_utils import torch_wav2spec

"""
@hanoihantrakul 4JUL2024 Implementation Notes for `umm_mkii_tts_rope.py`

At the time of writing, we were trying to improve ConformerUMM by revisiting
the ROPE implementation. In this implementation, I am using `TTS_ROPE` which is 
a modified ROPE implementation that is not the same as the orignal mkii version
and not the same as the mk4 version.

I decided to split this Stage2 into its own class because of the following reasons:

1.) Default BigMusic umm_mkii.Stage2 does not train on pitch related losses. 
It only trains on CTC, Chroma and Mel Spec. In SpeechTTS Stage2 however, 
the `add_pitch` flag will be turned on and RMVPE is used as a supervised 
f0_hz and vuv signal to compute a pitch loss.

I want to introduce a new pitch detect which is trained in-house on vocal music
data. I think it would be confusing to add another `add_vocal_pitch` ontop
of the TTS `add_pitch`. People would not know the difference and could
accidentally turn on both.

2.) Stage2 and Stage3 Training by default computes the Mel Spec loss on a
mel-128 spectrogram. The pitch detector however, requires a mel-160 spectrogram. 
This is due to historical reasons, because the pitch detector was originally
intended for DualUMM system, which operates on mel-160 spectrograms and not mel-128.

So, the Mel Reconstruction head will operate on mel-128 spectrograms. In addition
the preprocessing step will also include a mel-160 spectrogram so the trained
pitch predictor and can output a ground truth signal. 
"""


class Stage2TTSRope(Stage2):
    def __init__(self, config):
        """The init function will contain both elements of umm_mkii.py and umm_mkii_pitch.py"""

        """@hanoihantrakul: The following settings are identical to umm_mkii.Stage2()"""
        super().__init__(config)
        self.mel_head = Conv2dUpsampling(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True)
        )
        if config.get("add_ctc", True):
            self.ctc_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.add_chroma:
            self.chroma_transform = ChromaSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                n_chroma=config.n_chroma,
                normalized=False,
            )
            self.chroma_head = Conv2dUpsampling(
                config.hidden_size, config.n_chroma, use_bn=config.get("use_bn", True)
            )

        """@hanoihantrakul: The following logic is closer to umm_mkii_pitch.py"""
        if config.get("add_pitch", False):
            """
            Use in house pitch predictor for supervised loss in BigMusic pipeline.
            This is different from TTS, which uses RVMPE as supervised loss.
            """
            # This is the mel-160 transform for the pre-trained pitch detector.
            self.mel_160_transform = lambda x: torch_wav2spec(
                x, num_mels=config.n_mels_tgt, sample_rate=config.sample_rate
            )
            # the pl_module handles loading the pretrained state_dict of perceptual pitch predictor
            self.pitch_predictor = PerceptualPitchPredictor()
            # This head reconstructs the f0_hz and vuv signals from a mel-160 spectrogram
            self.f0_vuv_head = Conv2dUpsampling(config.hidden_size, 2)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        # This is mel-128 for computing the tokenizer's reconstruction loss
        normalize = self.config.feature_cmvn is not None
        if self.config.get("fix_stats_not_loading", False):
            """
            @hanoih 7JUL2024
            I discovered the weirdest bug where in SSTK training (not Vocal training), the self.audio_transform
            internal state will get reset. Eventhough the constructor of Stage2() correctly initiates the
            object with the dataset statistics for normalization, the `mel = self.audio_transform(x, normalize)`
            in the next line will fail unless I force self.audio_transform to load the stats again at the beginning of `self.preprocessing()`
            To fix this problem for now, I create this flag `fix_stats_not_loading` which is turned on only for
            SSTK training with TTS-ROPE implementation. I did not want to modify the original Stage2.preprocessing() code.
            It's a bizarre bug which I couldn't track down.
            """
            self.audio_transform.load_from_checkpoint(
                self.config.feature_cmvn
            )  # for some reason I have to do this everytime. If I leave it in the constructor it doesn't work for SSTK data (?)
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}

        # @hanoihantrakul: By default, this `interfere_audio` branch is never used by BigMusic not BigTTS.
        # if self.config.get("interfere_audio", None):
        #     x_interfered = self.interfere_audio(x)
        #     mel_interfered = self.audio_transform(x_interfered, normalize=normalize)
        #     input_dict.update(mel_interfered=mel_interfered)
        if self.config.add_chroma:
            chroma = self.chroma_transform(x)[:, :, :-1].transpose(1, 2)
            chroma = F.normalize(chroma, p=2, dim=-1)
            input_dict.update(chroma=chroma)
        if self.config.get("add_pitch", False):
            """
            @hanoihantrakul 4JUL2024 See umm_mkii_pitch.py on why this function is implemented
            differently than the TTS version.
            First, we use our own in-house trained pitch predictor and not an open source RVMPE
            - Originally, the function f0_normalize(f0) centers the signal with mean=0 and std=1 since TTS only cares about relative deviations in pitch.
            We do not want this in vocal music.
            - The function get_vuv() outputs a binary state. I apply the same logic to our in-house pitch predictor.
            """
            # This is mel-160 required by the pre-trained pitch detector
            mel_160 = self.mel_160_transform(x)
            input_dict.update(mel_160=mel_160)

            # This is pitch_detector logic.
            pd_output = self.pitch_predictor.forward(
                input_dict["mel_160"]
            )  # use mel_160 signal
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

        return input_dict

    def forward(self, input_dict):
        """
        @hanoihantrakul: The following logic is identical umm_mkii.Stage2.forward()
        """
        # @hanoih: "mel_interfered" is historical and never used
        # feature = (
        #     input_dict["mel_interfered"]
        #     if self.config.interfere_audio
        #     else input_dict["mel"]
        # )
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)
        output_dict = {"mel_out": mel_out, "flops": flops * 3}  # extra 2x for backward.
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            """
            @hanoihantrakul: The following settings are taken from umm_mkii_pitch.py when handling
            the supervised pitch detector outputs.
            """
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            f0_out = f0_vuv_out[:, :, 0:1]
            vuv_out = f0_vuv_out[:, :, 1:]
            # sometimes f0_gt, vuv_gt is 1 timestep longer than f0_out, vuv_out
            f0_vuv_trim_len = pitch_utils.compute_min_lengths(
                f0_out, input_dict["f0"], axis=1
            )
            input_dict["f0"] = input_dict["f0"][:, :f0_vuv_trim_len, :]
            f0_out = f0_out[:, :f0_vuv_trim_len, :]
            input_dict["vuv"] = input_dict["vuv"][:, :f0_vuv_trim_len, :]
            vuv_out = vuv_out[:, :f0_vuv_trim_len, :]
            output_dict.update(f0_out=f0_out)  # [batch_size, time_steps, 1]
            output_dict.update(vuv_out=vuv_out)  # [batch_size, time_steps, 1]
        return output_dict
