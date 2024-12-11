"""
@hanoihantrakul 4NOV2024 Implementation Notes for `umm_mkii_tts_rope_lfr.py`

LFR stands for "Low Frame Rate". 

The problem with the original UMM codebase was the pipeline
hard coded 100hz mel feature rate and 4x downsampling to 25hz token rate. When we wanted to try
20, 15 and 10 token rates, introducing this change into umm_mkii.py was not possible withoout
re writing a few modules.

At the same time, Ju Chiang was working on UMM2 and performing regression tests so we could migrate
from the relatively messy UMM codebase to the cleaner and modularized UMM2. At the time of writing,
we are still regression testing UMM2.

Therefore, I have implemented the LFR version of the ConformerUMM_TTS_ROPE tokenizer using the 
older UMM codebase. Once UMM2 has been properly tested E2E, we can migrate this implementation
into UMM2. I've extensively commented the code to make migration easier. 
"""


import torch
from torch import Tensor, int32, nn
from torch.nn import functional as F

from recipes.umm.models.dualumm_vector_quantizers import (
    get_vector_quantizer,
    get_vector_quantizer_projection_layers,
    get_vq_codebook_distances,
)
from recipes.umm.models.voc_modules.pitch_predictor import pitch_utils
from recipes.umm.models.voc_modules.pitch_predictor.inference import (
    PerceptualPitchPredictor,
)
from einops import pack, rearrange
from recipes.umm.transforms.chroma import ChromaSpectrogram
from recipes.umm.utils.mel_utils import torch_wav2spec
from recipes.umm.models.umm_mkii import RandomProjectionQuantizer, Base, ConformerEncoderLayer, Stage2


class Conv2dSubsamplingLFR(nn.Module):
    """
    @hanoihantrakul 4NOV2024
    The original umm_mkii.Conv2dSubsampling hardcoded 4x downsampling assuming the input was 100hz. 
    In this implementation I support downsampling from 120Hz to 20, 15 and 10 Hz. I had to rewrite the striding
    logic and input/output shape logic. 
    """
    def __init__(
        self, input_dim, output_dim, kernel, padding, use_bn=True, act_fn=nn.ReLU, conv_config_dict=None,
    ):
        super().__init__()        
        if conv_config_dict is None:
            # Default to 15hz case
            conv_config_dict = {'conv1': 2, 'conv2': 2, 'conv3': 2, 'linear': 8192} # Why 8192? -> 128/2/2/2 = 16 which becomes 16 after all the conv ops on mel-128. Then 16*512 (last layer channels) gives 8096 total dims to be projected down to output_dim=1024

        """
        @hanoihantrakul 4NOV2024
        Yes, you can do the following in a for loop. However, when I was re-writing
        the UMM library to support downsampling to 20,15,10 hz there were a lot of changes
        to the striding patterns required. Notably, you need 3 conv layers to achieve smooth
        downsampling instead of 2 conv layers. For this reason, I wrote the code like this
        to make this logic explicit and make integration in UMM2 much easier. 

        Basically, the previous implementation had 2 layers of conv with kernel size 5.
        If you try to do 10x downsampling by doing stride=2 and then stride=5, you will be
        jumping over too many features in the second layer when going from 100hz -> 10hz. A stride of 5 and kernel size of 5
        means there is no overlap in the features, which will make the implementation not comparable to the previous tokenizer.

        Instead I did this by making the input feature 120Hz and then downsampling by strides 2, then 2 and then 3.
        120/2/2/3 = 10hz. This ensures we still have overlapping regions with kernel 5.
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel, conv_config_dict['conv1'], padding),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            act_fn(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel, conv_config_dict['conv2'], padding),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            act_fn(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel, conv_config_dict['conv3'], padding),
            nn.BatchNorm2d(512) if use_bn else nn.Identity(),
            act_fn(),
        )

        self.linear = nn.Linear(conv_config_dict['linear'], output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x

    # def get_flops(self, b, t, d):
    #     flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
    #     flops2, _ = conv_flops(self.conv[3], out_shape1)
    #     return flops1 + flops2
    
    def get_flops(self, b, t, d):
        """TODO: Properly calculate the num_flops"""
        return 0

class Conv2dUpsamplingLFR(nn.Module):
    """
    @hanoihantrakul 4NOV2024
    The original umm_mkii.Conv2dUpsampling hardcoded 4x upsampling assuming the input was originally 100hz. 
    In this implementation I support downsampling from 120Hz to 20, 15 and 10 Hz. I had to rewrite the striding
    logic and input/output shape logic. 
    """
    def __init__(self, input_dim, output_dim, use_bn=True, act_fn=nn.ReLU, conv_config_dict=None):
        super().__init__()

        if conv_config_dict is None:
            # Default to 15hz case
            conv_config_dict = {'conv1': 2, 'conv2': 2, 'conv3': 2, 'linear': 8192} # Why 8192? -> 128/2/2/2 = 16 which becomes 16 after all the conv ops on mel-128. Then 16*512 (last layer channels) gives 8096 total dims to be projected down to output_dim=1024

        """
        @hanoihantrakul 4NOV2024
        Yes, you can do the following in a for loop. However, when I was re-writing
        the UMM library to support downsampling to 20,15,10 hz there were a lot of changes
        to the striding patterns required. Notably, you need 1 conv layer and 3 upconv layers to achieve smooth
        downsampling instead of 1 conv layer and 2 upconv layers. For this reason, I wrote the code like this
        to make this logic explicit and make integration in UMM2 much easier. 
        """
        self.conv =  nn.Sequential(
            nn.Conv2d(1, 64, 7, 1, 3),
            torch.nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            act_fn())

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 6, conv_config_dict['conv1'], 2),
            torch.nn.BatchNorm2d(16) if use_bn else nn.Identity(),
            act_fn())
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 6, conv_config_dict['conv2'], 2),
            torch.nn.BatchNorm2d(8) if use_bn else nn.Identity(),
            act_fn())
        
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, 6, conv_config_dict['conv3'], 2),
            torch.nn.BatchNorm2d(1) if use_bn else nn.Identity(),
            act_fn())

        self.linear = nn.Linear(conv_config_dict['linear'], output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x

    # def get_flops(self, b, t, d):
    #     flops1, out_shape1 = conv_flops(self.conv[0], [b, 1, t, d])
    #     flops2, out_shape2 = conv_transpose_flops(self.conv[3], out_shape1)
    #     flops3, _ = conv_transpose_flops(self.conv[6], out_shape2)
    #     return flops1 + flops2 + flops3

    def get_flops(self, b, t, d):
        """TODO: Properly calculate the num_flops"""
        return 0
    

class AudioEncoderLFR(nn.Module):
    """
    @hanoihantrakul 4NOV2024
    Main difference with umm_mkii.AudioEncoder() is how the underlying
    Conv2DSubsamplingLFR is not hard coded to 100hz mel features and 4x downsampling to 25hz.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_encoder = Conv2dSubsamplingLFR(
            config.num_channels,
            config.hidden_size,
            config.feature_encoder_kernel,
            config.feature_encoder_padding,
            use_bn=config.get("use_bn", True),
            conv_config_dict=config.get_conv_downsampling_config(),
        )
        self.conformer_layer = (
            ConformerEncoderLayer(config)
            if config.get("first_conformer", True)
            else nn.Identity()
        )

    def forward(self, x):
        x = self.feature_encoder(x)
        x = self.conformer_layer(x)
        return x

    def get_flops(self, b, t, d):
        if self.config.get("first_conformer", True):
            return self.feature_encoder.get_flops(
                b, t, d
            ) + self.conformer_layer.get_flops(b, t)
        else:
            return self.feature_encoder.get_flops(b, t, d)
        

class Stage1LFR(Base):
    '''
    @hanoihantrakul 4NOV2024 
    Stage1LFR could be used for TTS_ROPE and non TTS_ROPE implementations. 
    '''
    def __init__(self, config):
        super().__init__(config)
        if config.rq_input_layernorm:
            self.rq_input_layernorm = nn.LayerNorm(
                config.num_channels
                * pow(config.feature_encoder_kernel, config.feature_encoder_padding),
                elementwise_affine=False,
            )
        self.rq = RandomProjectionQuantizer(config)
        self.rq_head = nn.Linear(
            config.hidden_size,
            config.rq_codebook_size * config.rq_codebook_num,
            bias=False,
        )
        '''
        @hanoihantrakul 4NOV2024 
        Remove the default self.audio_encoder = AudioEncoder(config) which was hardcoded to 4x downsampling for a 25hz target token rate.
        '''
        del self.audio_encoder
        self.audio_encoder = AudioEncoderLFR(config) # this one supports 20,15 and 10Hz target token rate

        '''
        @hanoihantrakul 4NOV2024
        Yes, you can configure the following Unfolders() with a for loop. However, when I was reverse-engineering
        this part of the code to support 20, 15 and 10 Hz sampling rate, I found it is much easier
        to write this code as 3 separate Unfold() operations. This is because, the Unfold() operation
        is like a manual version of the Conv2D operation. If Stage2 and Stage3 uses 3 layers of Conv2D 
        downsampling and then 3 layers of Conv2D Upsampling, then there must be 3 Unfolders in Stage1. 
        If there is a mismatch, you get some very annoying CUDA-level errors. 
        Writing it this way makes it explicit the relationship between the Unfolder, Conv2D and UpConv2D.

        I recommend https://mrinath.medium.com/vit-part-1-patchify-images-using-pytorch-unfold-716cd4fd4ef6 
        to understand the underlying operations and output shapes of the Unfold() operation. It was more
        helpful than the pytorch documentation (https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)
        '''
        self.conv_config_dict = config.get_conv_downsampling_config()
        self.unfolder1 = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(self.conv_config_dict['conv1'], 1),
        )
        self.unfolder2 = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(self.conv_config_dict['conv2'], 1),
        )
        self.unfolder3 = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(self.conv_config_dict['conv3'], 1),
        )

    def forward(self, input_dict):
        """
        @hanoihantrakul 4NOV2024
        Reverse engineering this function was confusing without the original author explaining it to me.
        I have added shape as comments because they were key in helping me figure out
        the shapes so I could downsample the features to 20,15 and 10 Hz instead of the hard coded 25hz. 

        Essentially, Stage2 and Stage3 use Conv2D operations to subsample the input. But in Stage1, these
        operations are replaced with Unfold() which allows manual control over how the input is downsampled and
        fed into the random quantizer. 
        """
        ### Model Outputs ###
        masked_feature = input_dict["masked_mel"] # [9, 3945, 128] [batch_size, time_steps, num_mels]
        masked_indices = input_dict["masked_indices"] # [1128, 2] [length_of_mask]
        flops = self.audio_encoder.get_flops(*masked_feature.shape)
        encoded_masked_feature = self.audio_encoder(masked_feature)
        hidden_states = self.encoder_input_dropout(encoded_masked_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        flops += (
            hidden_states.shape[0]
            * hidden_states.shape[1]
            * self.rq_head.weight.shape[0]
            * self.rq_head.weight.shape[1]
            * 2
        )
        # At this point hidden_states = [9, 658, 1024] [batch_size, time_steps, hidden_size]
        # 3945 -> 658 is 6x downsampling (rounded to nearest int). 120Hz -> 6x downsampling to 20Hz.
        logits = self.rq_head(hidden_states)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        ) # [9, 658, 4096, 8] [batch_size, time_steps, codebook_size, codebook_num] This represents the logits for each code selected from random projection codebook (BEST-RQ)

        
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        ) # [9024, 4096] 

        ### Generated Masked Features for Random Projection ###
        feature = input_dict["mel"] # [9, 3945, 128] [batch_size, time_steps, num_mels]

        target = self.get_rq_target(feature) # See self.get_rq_target() for more shape comments
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "rq_logits": logits, # [9, 658, 4096, 8] 
            "rq_masked_logits": masked_logits, # [9024, 4096]
            "rq_target": target, # [9, 658, 4096]
            "rq_masked_target": masked_target, # [9024]
            "flops": flops * 3,  # extra 2x for backward.
        }
        return output_dict

    def _unfold(self, feature, unfolder):
        """This operation replaces a traditional Conv2D operation."""
        # feature.shape = [9, 3945, 128] [batch_size, time_steps, num_mels]
        d = feature.size(-1) # d=128
        unfold_feature = feature.unsqueeze(1) # unfold_feature.shape = [9, 1, 3945, 128]
        unfold_feature = unfolder(unfold_feature) # unfold_feature.shape = [9, 5, 504960]
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d) # unfold_feature.shape = [9, 3495, 640]
        return unfold_feature
    
    def _subsample(self, feature):
        # Assume input shape is [9, 3945, 128] for mel features being downsampled to 20hz with pattern [1,2,3]
        # feature.shape = [9, 3945, 128]
        feature = self._unfold(feature, unfolder=self.unfolder1) 
        # feature.shape = [9, 3945, 640]
        feature = self._unfold(feature, unfolder=self.unfolder2)
        # feature.shape = [9, 1973, 3200]
        feature = self._unfold(feature, unfolder=self.unfolder3)
        # feature.shape = [9, 658, 16000]
        '''
        @hanoihantrakul 4NOV2024
        So this operation has replicated the behavior of 3 conv layers with stride downsampling pattern [1,2,3].
        3945 -> 658 is 6x downsampling (rounded to nearest int). 120Hz -> 6x downsampling to 20Hz.
        '''
        return feature

    @torch.no_grad()
    def get_rq_target(self, feature):
        rq_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
        if self.config.rq_input_layernorm:
            rq_input = self.rq_input_layernorm(rq_input)
        # rq_input.shape = [5922, 16000] where 5922 = 9*658 from the application of 3 Unfolder() in self._subsample()
        target_tokens = self.rq(rq_input)
        # target_tokens = [5922, 8]
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=feature.size(0))
        # target_tokens.shape = [9, 658, 8] which is the expected shape for the ground truth tokens.
        return target_tokens
    

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        return {"mel": mel}
    

class Stage2TTSRopeLFR(Base):
    def __init__(self, config):
        """
        @hanoihantrakul 5NOV2024
        This implementation is similar to umm_mkii_tts_rope.Stage2TTSROPE(), but with new logic for handling 
        120hz mel feature inputs.

        It will be properly refacted into UMM2 when regression testing of UMM2 is complete. I've copy pasted 
        a lot of code from umm_mkii_tts_rope.Stage2TTSROPE() to speed up spiking. 
        """

        super().__init__(config)
        '''
        @hanoihantrakul 4NOV2024 
        Remove the default self.audio_encoder = AudioEncoder(config) which was hardcoded to 4x downsampling for a 25hz target token rate.
        '''
        del self.audio_encoder
        self.audio_encoder = AudioEncoderLFR(config) # this one supports 20,15 and 10Hz target token rate

        # Add reconstruction heads which support mel features at 120hz and downsampling/upsampling to 20,15,10 token rate
        self.mel_head = Conv2dUpsamplingLFR(
            config.hidden_size, config.n_mels, use_bn=config.get("use_bn", True), conv_config_dict=config.get_conv_upsampling_config()
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
            self.chroma_head = Conv2dUpsamplingLFR(
                config.hidden_size, config.n_chroma, use_bn=config.get("use_bn", True), conv_config_dict=config.get_conv_upsampling_config()
            )

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
            self.f0_vuv_head = Conv2dUpsamplingLFR(config.hidden_size, 2, conv_config_dict=config.get_conv_upsampling_config())

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        # This is mel-128 for computing the tokenizer's reconstruction loss
        normalize = self.config.feature_cmvn is not None
        mel = self.audio_transform(x, normalize=normalize)
        input_dict = {"mel": mel}

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

        # Process Mel
        """
        @hanoihantrakul 12JUL2024
        I copied this code from `umm_mkii_pitch` to handle the new f0_hz signal.

        `mel` (ground truth) can sometimes be 1 sample longer than `mel_out` (predicted)
        - Just for this one config only, correct for this 1 sample difference.
        - e.g. [6, 2917, 160] vs [6, 2916, 160]
        - This will not be a problem if a compeletely new class is defined without inheritance from Stage1 and Stage2
        """
        mel_out = self.mel_head(hidden_states)
        mel_trim_len = pitch_utils.compute_min_lengths(
            input_dict["mel"], mel_out, tolerance=15, axis=1
        )
        input_dict["mel"] = input_dict["mel"][:, :mel_trim_len, :]
        mel_out = mel_out[:, :mel_trim_len, :]

        output_dict = {"mel_out": mel_out, "flops": flops * 3}  # extra 2x for backward.
        if self.config.get("add_ctc", True):
            flops += 2 * torch.numel(hidden_states) * self.ctc_head.weight.shape[0]
            ctc_out = self.ctc_head(hidden_states)
            output_dict.update(ctc_out=ctc_out)
        if self.config.add_chroma:
            chroma_out = self.chroma_head(hidden_states)
            chroma_trim_len = pitch_utils.compute_min_lengths(
                chroma_out, input_dict['chroma'], axis=1, tolerance=15
            )
            input_dict["chroma"] = input_dict["chroma"][:, :chroma_trim_len, :]
            chroma_out = chroma_out[:, :chroma_trim_len, :]
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            f0_out = f0_vuv_out[:, :, 0:1]
            vuv_out = f0_vuv_out[:, :, 1:]
            '''
            30OCT2024 @hanoihantrakul
            The original pitch detector was trained to output features at 100hz. This tokenizer expects 120hz features. 
            The only thing we can do here is either train a new pitch detector to output 120hz features.
            or resample (interpolate) a 100hz signal to 120hz. 
            I choose to use interpolation because this is a low frequency control signal, not an audio-rate signal. 
            An anti-aliasing filter is not required to interpolate 100hz pitch signal to 120hz.            
            '''
            f0_gt, vuv_gt = input_dict['f0'], input_dict['vuv']
            # Need to transpose [batch_size, time_steps, 1] to [batch_size, 1, time_steps] since the interpolate function needs dimension to be interpolated at the end.
            f0_gt_interp = F.interpolate(f0_gt.transpose(1,2), size=f0_out.shape[1], mode="linear")
            vuv_gt_interp = F.interpolate(vuv_gt.transpose(1,2), size=f0_out.shape[1], mode="linear")
            input_dict['f0'], input_dict['vuv'] = f0_gt_interp.transpose(1,2), vuv_gt_interp.transpose(1,2)

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
    


class Stage3TTSRopeLFR(Stage2TTSRopeLFR):
    """
    @hanoihantrakul 4NOV2024
    As mentioned before, this contains a lot of copy pasted code from Stage2TTSRopeLFR and 
    umm_mkii_tts_rope.Stage3TTSRope(). It will be refactored neatly into UMM2 when
    regression testing is complete. 
    """
    def __init__(self, config):
        super().__init__(config)
        self.init_vq_layers(config)

    def init_vq_layers(self, config):
        """
        Init the VQ layer.
        @hanoihantrakul: I copied this from `convumm_gan.py` because it is cleaner than the original umm_mkii.Stage3 code
        """
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

    def forward(self, input_dict):
        """
        12JUL2024 @hanoih:
        This function is unfortunately becoming very large since it combines code from
        - umm_mkii.stage3.forward()
        - umm_mkii_pitch.stage3.forward()
        - convumm_gan.forward()

        If this approach works, recommend refactoring to a cleaner implementation.
        """
        feature = input_dict["mel"]
        flops = self.audio_encoder.get_flops(*feature.shape)
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        flops += self.encoder_layers[0].get_flops(*hidden_states.shape[0:2]) * len(
            self.encoder_layers
        )

        """
        @hanoihantrakul 12JUL2024
        I copied this part of the code from `umm_mkii.Stage3 forward`.
        It can be simplified to be more like `convumm_gan.forward_vq()`
        """
        for i, layer in enumerate(self.encoder_layers):
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
                    vq_embs, vq_ids, vq_loss = self.vq(hidden_states)
                # calculate codebook distances by accessing the VQ's internal matrix representing the actual codebook
                codebook_distance_stats = get_vq_codebook_distances(
                    self.vq.embedding.weight.data
                )
                hidden_states = self.vq_proj_out(vq_embs)
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )

        flops += self.mel_head.get_flops(*hidden_states.shape)
        mel_out = self.mel_head(hidden_states)

        """
        @hanoihantrakul 12JUL2024
        I copied this code from `umm_mkii_pitch` to handle the new f0_hz signal.

        `mel` (ground truth) can sometimes be 1 sample longer than `mel_out` (predicted)
        - Just for this one config only, correct for this 1 sample difference.
        - e.g. [6, 2917, 160] vs [6, 2916, 160]
        - This will not be a problem if a compeletely new class is defined without inheritance from Stage1 and Stage2
        """
        mel_trim_len = pitch_utils.compute_min_lengths(
            input_dict["mel"], mel_out, tolerance=15, axis=1
        )
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
        if self.config.get("add_chroma", False):
            chroma_out = self.chroma_head(hidden_states)
            chroma_trim_len = pitch_utils.compute_min_lengths(
                chroma_out, input_dict['chroma'], axis=1, tolerance=15
            )
            input_dict["chroma"] = input_dict["chroma"][:, :chroma_trim_len, :]
            chroma_out = chroma_out[:, :chroma_trim_len, :]
            output_dict.update(chroma_out=chroma_out)
        if self.config.get("add_pitch", False):
            f0_vuv_out = self.f0_vuv_head(hidden_states)
            f0_out = f0_vuv_out[:, :, 0:1]
            vuv_out = f0_vuv_out[:, :, 1:]
            '''
            30OCT2024 @hanoihantrakul
            The original pitch detector was trained to output features at 100hz. This tokenizer expects 120hz features. 
            The only thing we can do here is either train a new pitch detector to output 120hz features.
            or resample (interpolate) a 100hz signal to 120hz. 
            I choose to use interpolation because this is a low frequency control signal, not an audio-rate signal.            
            '''
            f0_gt, vuv_gt = input_dict['f0'], input_dict['vuv']
            # Need to transpose [batch_size, time_steps, 1] to [batch_size, 1, time_steps] since the interpolate function needs dimension to be interpolated at the end.
            f0_gt_interp = F.interpolate(f0_gt.transpose(1,2), size=f0_out.shape[1], mode="linear")
            vuv_gt_interp = F.interpolate(vuv_gt.transpose(1,2), size=f0_out.shape[1], mode="linear")
            input_dict['f0'], input_dict['vuv'] = f0_gt_interp.transpose(1,2), vuv_gt_interp.transpose(1,2)
            
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
        # add the codebook stats
        output_dict.update(codebook_distance_stats)
        return output_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _prepare_wav(self, wav):
        """Check audio dimensions and pad."""
        if wav.dim() == 3:
            wav = wav.squeeze(dim=1)
        return self.pad_audio(wav.float())

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _get_vq_ids(self, hidden_states, position_embeddings):
        """Apply Vector Quantization and only get the ID's."""
        for i, layer in enumerate(self.encoder_layers):
            if i == self.config.vq_layer_idx:
                vq_hidden_states = self.vq_proj_in(hidden_states)
                _, vq_ids, _ = self.vq(vq_hidden_states)
                return {"vq_ids": vq_ids, "hidden_states": hidden_states}
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return {"vq_ids": vq_ids, "hidden_states": hidden_states}

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """Convert audio file to tokens (after Vector Quantization)."""
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        result = self._get_vq_ids(hidden_states, position_embeddings)
        return result["vq_ids"]
    

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav):
        """Convert audio file to tokens (after Vector Quantization)."""
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        result = self._get_vq_ids(hidden_states, position_embeddings)
        return result["vq_ids"]

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _get_pre_vq_latents(self, hidden_states, position_embeddings):
        """Apply Vector Quantization and only get the ID's."""
        if self.config.vq_layer_idx > 0:
            for i, layer in enumerate(self.encoder_layers):
                if i == self.config.vq_layer_idx:
                    vq_hidden_states = self.vq_proj_in(hidden_states)
                    # _, vq_ids, _ = self.vq(vq_hidden_states)
                    return vq_hidden_states
                hidden_states = layer(
                    hidden_states, position_embeddings=position_embeddings
                )
        else:
            raise ValueError("Model does not have a VQ layer")

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2pre_vq_latents(self, wav):
        """Convert audio file to continuous latents (before Vector Quantization)."""
        wav = self._prepare_wav(wav)
        feature = self.preprocessing(wav)["mel"]
        audio_feature = self.audio_encoder(feature)
        hidden_states = self.encoder_input_dropout(audio_feature)
        position_embeddings = self.embed_positions(hidden_states)
        return self._get_pre_vq_latents(hidden_states, position_embeddings)