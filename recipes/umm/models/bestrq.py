import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torchaudio.transforms import AmplitudeToDB

from recipes.umm.models.flash_conformer import Wav2Vec2ConformerEncoder
from recipes.umm.models.vocoder import BigVGAN
from recipes.umm.models.vq import VectorQuantize
from recipes.umm.transforms.speech import SpeechTransform
from samantha.transforms.audio import MelSpectrogram
from samantha.utils.hparams import DotDict


class Conv2dUpsampling(nn.Module):
    def __init__(self, input_dim, output_dim, up_scale=4):
        super().__init__()
        assert up_scale in [2, 4]
        if up_scale == 2:
            self.conv = nn.Sequential(
                # [1, 1, 750, 1024]
                nn.Conv2d(1, 32, 7, 1, 3),
                torch.nn.BatchNorm2d(32),
                nn.ReLU(),
                # [1, 32, 750, 1024]
                nn.ConvTranspose2d(32, 1, 6, 2, 2),
                torch.nn.BatchNorm2d(1),
                nn.ReLU(),
                # [1, 1, 1500, 2048]
            )
        elif up_scale == 4:
            self.conv = nn.Sequential(
                # [1, 1, 750, 1024]
                nn.Conv2d(1, 64, 7, 1, 3),
                torch.nn.BatchNorm2d(64),
                nn.ReLU(),
                # [1, 64, 750, 1024]
                nn.ConvTranspose2d(64, 8, 6, 2, 2),
                torch.nn.BatchNorm2d(8),
                nn.ReLU(),
                # [1, 8, 1500, 2048]
                nn.ConvTranspose2d(8, 1, 6, 2, 2),
                torch.nn.BatchNorm2d(1),
                nn.ReLU(),
                # [1, 1, 3000, 4096]
            )
        conv_out_dim = input_dim * up_scale
        self.linear = nn.Linear(conv_out_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x


class Conv2dSubsampling(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, conv_layers):
        super().__init__()
        assert len(conv_layers) in [1, 2]
        modules = []
        prev_c = 1
        conv_out_dim = conv_layers[-1] * input_dim
        for c in conv_layers:
            modules.append(nn.Conv2d(prev_c, c, kernel, 2, kernel // 2))
            modules.append(nn.ReLU())
            prev_c = c
            conv_out_dim //= 2
        self.conv = nn.Sequential(*modules)
        self.linear = nn.Linear(conv_out_dim, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        x = rearrange(x, "b c t f -> b t (c f)")
        x = self.linear(x)
        return x


class RandomProjectionQuantizer(nn.Module):
    """RandomProjectionQuantizer"""

    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        quantizer_num=1,
        prototype_initialization_method="gaussian",
        projection_initialization_method="xavier",
    ):
        """A quantizer based on random projection
        See: https://arxiv.org/pdf/2202.01855.pdf
        Args:
            dim: input dimension (channels)
            codebook_size: the number of code in the codebook
            codebook_dim: the dimension of the the code
            quantizer_num: the number of quantizers.
                    See multi-softmax in https://arxiv.org/abs/2303.01037
            initialization_type: the initialization method of the projection matrix
        """
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_num = quantizer_num
        self.prototype_initialization_method = prototype_initialization_method
        self.projection_initialization_method = projection_initialization_method

        self.register_buffer(
            "prototypes",
            torch.zeros(
                self.quantizer_num,
                1,
                self.codebook_size,
                self.codebook_dim,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "proj",
            torch.zeros(
                self.input_dim,
                self.quantizer_num * self.codebook_dim,
                requires_grad=False,
            ),
        )
        self._initialize()

    def _initialize(self):
        """Initialize the parameters."""
        if self.prototype_initialization_method == "gaussian":
            nn.init.normal_(self.prototypes)
            F.normalize(self.prototypes, dim=-1, out=self.prototypes)
        else:
            raise ValueError(
                "Unknown prototype_initialization_method:"
                " {}".format(self.prototype_initialization_method)
            )

        if self.projection_initialization_method == "xavier":
            fan_in = self.input_dim
            fan_out = self.codebook_dim
            gain = 1.0
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            with torch.no_grad():
                self.proj.normal_(0, std)
        else:
            raise ValueError(
                "Unknown projection_initialization_method:"
                " {}".format(self.projection_initialization_method)
            )

    def forward(self, x):
        """
        Forward a batch of representations to get discrete codes.
        Args:
            x: [batch_size, dim]
        Returns:
            codes: [batch_size, quantizer_num], torch.int64
        """
        assert len(x.shape) == 2
        batch_size, _ = x.shape

        projected = torch.matmul(
            x, self.proj
        )  # [batch_size, quantizer_num*codebook_dim]
        projected = F.normalize(
            projected.view(batch_size, self.quantizer_num, self.codebook_dim),
            p=2,
            dim=-1,
        )  # [batch_size, quantizer_num, codebook_dim]
        projected = projected.permute(1, 0, 2).view(
            self.quantizer_num, batch_size, 1, self.codebook_dim
        )  # [quantizer_num, batch_size, 1, codebook_dim]

        # self.prototypes: [quantizer_num, 1, codebook_size, codebook_dim]
        # it is normalized in the function _initialize

        # TODO: configure multiple distances, such as cosine similarity
        # distances = torch.norm(projected - self.prototypes, p=2, dim=-1)
        # [quantizer_num, batch_size, codebook_size]

        # Save spaces.
        distances = (
            projected.view(self.quantizer_num, batch_size, self.codebook_dim)
            .pow(2)
            .sum(-1, keepdim=True)  # [quantizer_num, batch_size, 1]
            + self.prototypes.view(
                self.quantizer_num, self.codebook_size, self.codebook_dim
            )
            .pow(2)
            .sum(-1, keepdim=True)
            .transpose(1, 2)  # [quantizer_num, 1, codebook_size]
            - 2
            * torch.bmm(
                projected.view(self.quantizer_num, batch_size, self.codebook_dim),
                self.prototypes.view(
                    self.quantizer_num, self.codebook_size, self.codebook_dim
                ).transpose(1, 2),
            )  # [quantizer_num, batch_size, codebook_size]
        )

        codes = torch.argmin(distances, dim=-1).transpose(0, 1)
        # [batch_size, quantizer_num]
        return codes


class BestRQ(nn.Module):
    def __init__(
        self, frontend_config, encoder_config, codebook_config, vq_config=None
    ):
        super().__init__()
        frontend_config = DotDict(frontend_config)
        codebook_config = DotDict(codebook_config)
        if vq_config is not None:
            vq_config = DotDict(vq_config)
            self.vq = VectorQuantize(**vq_config)
            self.vq_config = vq_config
        else:
            self.vq = None

        self.frontend = Conv2dSubsampling(
            input_dim=frontend_config.n_mels,
            output_dim=encoder_config.hidden_size,
            kernel=frontend_config.kernel,
            conv_layers=frontend_config.conv_dim,
        )
        if frontend_config.get("quantizer_layernorm", True):
            self.quantizer_layernorm = nn.LayerNorm(
                frontend_config.n_mels
                * pow(frontend_config.kernel, len(frontend_config.conv_dim)),
                elementwise_affine=False,
            )

        self.encoder = Wav2Vec2ConformerEncoder(encoder_config)
        self.codebook_head = nn.Linear(
            encoder_config.hidden_size,
            codebook_config.codebook_size * codebook_config.n_softmax,
            bias=False,
        )

        self.model_input_transform = SpeechTransform(
            sample_rate=frontend_config.sample_rate,
            n_mels=frontend_config.n_mels,
            n_fft=frontend_config.n_fft,
            win_length=frontend_config.win_length,
            hop_length=frontend_config.hop_length,
            f_min=0,
            f_max=frontend_config.sample_rate // 2,
        )
        if frontend_config.feature_cmvn is not None:
            self.model_input_transform.load_from_checkpoint(
                frontend_config.feature_cmvn
            )

        self.unfolder = nn.Unfold(
            kernel_size=(frontend_config.kernel, 1),
            dilation=1,
            padding=(frontend_config.kernel // 2, 0),
            stride=(2, 1),
        )
        self.quantizer = RandomProjectionQuantizer(
            input_dim=frontend_config.n_mels
            * pow(frontend_config.kernel, len(frontend_config.conv_dim)),
            codebook_dim=codebook_config.codebook_dim,
            codebook_size=codebook_config.codebook_size,
            quantizer_num=codebook_config.n_softmax,
        )

        self.frontend_config = frontend_config
        self.encoder_config = encoder_config
        self.codebook_config = codebook_config

    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        for _ in range(len(self.frontend_config.conv_dim)):
            feature = self._unfold(feature)
        return feature
    @torch.no_grad()
    def get_target_tokens(self, feature):
        quantizer_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
        if self.frontend_config.get("quantizer_layernorm", True):
            quantizer_input = self.quantizer_layernorm(quantizer_input)
        # get target tokens
        target_tokens = self.quantizer(quantizer_input)
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=feature.size(0))
        return target_tokens

    def forward(self, masked_feature, masked_indices, feature):
        encoded_masked_feature = self.frontend(masked_feature)
        if self.vq is not None:
            model_output = self.encoder(encoded_masked_feature, vq=self.vq)
            hidden_state = model_output["last_hidden_state"]
            quant_encoder_out, quant_idx, quant_loss = model_output["quant_state"]
        else:
            hidden_state = self.encoder(encoded_masked_feature)["last_hidden_state"]
        logits = self.codebook_head(hidden_state)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.codebook_config.n_softmax
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.codebook_config.n_softmax
        )
        target_tokens = self.get_target_tokens(feature)
        masked_target_tokens = target_tokens[tuple(masked_indices.t())]
        masked_target_tokens = rearrange(masked_target_tokens, "b c -> (b c)")
        if self.vq is not None:
            return (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            )
        else:
            return logits, masked_logits, target_tokens, masked_target_tokens

    def pad_audio_for_down_and_upsampling(self, x):
        down_scale = 2 ** len(self.frontend_config.conv_dim)
        if x.size(-1) % (self.frontend_config.hop_length * down_scale) > 0:
            return F.pad(
                x,
                (
                    0,
                    self.frontend_config.hop_length * down_scale
                    - (x.size(-1) % (self.frontend_config.hop_length * down_scale)),
                ),
                "constant",
                0,
            )
        else:
            return x

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, x):
        x = self.pad_audio_for_down_and_upsampling(x)
        normalize = self.frontend_config.feature_cmvn is not None
        return self.model_input_transform(x, normalize=normalize)

    def get_latent(self, x, layer_idx: Optional[int] = None):
        x = self.frontend(x)
        if layer_idx is None:
            emb = self.encoder(x, vq=self.vq)["last_hidden_state"]
            return emb
        else:
            emb = self.encoder(x, output_hidden_states=True, vq=self.vq)[
                "hidden_states"
            ]
            return emb[layer_idx]


class BestRQMel(BestRQ):
    def __init__(
        self, frontend_config, encoder_config, codebook_config, vq_config=None
    ):
        super().__init__(
            frontend_config=frontend_config,
            encoder_config=encoder_config,
            codebook_config=codebook_config,
            vq_config=vq_config,
        )
        self.spec_reconstructor = Conv2dUpsampling(
            self.encoder_config.hidden_size,
            self.frontend_config.n_mels,
            2 ** len(self.frontend_config.conv_dim),
        )
    def forward(self, masked_feature, masked_indices, feature):
        encoded_masked_feature = self.frontend(masked_feature)
        if self.vq is not None:
            model_output = self.encoder(encoded_masked_feature, vq=self.vq)
            hidden_state = model_output["last_hidden_state"]
            quant_encoder_out, quant_idx, quant_loss = model_output["quant_state"]
        else:
            hidden_state = self.encoder(encoded_masked_feature)["last_hidden_state"]
        recon_feature = self.spec_reconstructor(hidden_state)
        logits = self.codebook_head(hidden_state)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.codebook_config.n_softmax
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.codebook_config.n_softmax
        )
        target_tokens = self.get_target_tokens(feature)
        masked_target_tokens = target_tokens[tuple(masked_indices.t())]
        masked_target_tokens = rearrange(masked_target_tokens, "b c -> (b c)")
        if self.vq is not None:
            return (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                recon_feature,
                quant_encoder_out,
                quant_idx,
                quant_loss,
            )
        else:
            return (
                logits,
                masked_logits,
                target_tokens,
                masked_target_tokens,
                recon_feature,
            )


class BestRQMelCTC(BestRQ):
    def __init__(
        self, frontend_config, encoder_config, codebook_config, vq_config=None
    ):
        super().__init__(
            frontend_config=frontend_config,
            encoder_config=encoder_config,
            codebook_config=codebook_config,
            vq_config=vq_config,
        )
        del self.codebook_head
        self.spec_reconstructor = Conv2dUpsampling(
            self.encoder_config.hidden_size,
            self.frontend_config.n_mels,
            2 ** len(self.frontend_config.conv_dim),
        )
        self.lm_head = nn.Linear(
            self.encoder_config.hidden_size, self.encoder_config.vocab_size
        )

    def forward(self, feature):
        encoded_feature = self.frontend(feature)
        if self.vq is not None:
            model_output = self.encoder(encoded_feature, vq=self.vq)
            hidden_state = model_output["last_hidden_state"]
            quant_encoder_out, quant_idx, quant_loss = model_output["quant_state"]
        else:
            hidden_state = self.encoder(encoded_feature)["last_hidden_state"]
        recon_feature = self.spec_reconstructor(hidden_state)
        logits = self.lm_head(hidden_state)
        if self.vq is not None:
            return logits, recon_feature, quant_encoder_out, quant_idx, quant_loss
        else:
            return logits, recon_feature

    def token_to_hidden_state(self, tokens):
        assert self.vq is not None
        token_embeddings = self.vq.get_codes_from_indices(tokens.long()).view(*tokens.shape, -1)
        token_embeddings = self.vq.project_out(token_embeddings)
        hidden_state = self.encoder.forward_from_vq(token_embeddings)
        return hidden_state

    def token_to_mel(self, tokens):
        hidden_state = self.token_to_hidden_state(tokens)
        recon_feature = self.spec_reconstructor(hidden_state)
        return recon_feature
    
    def token_to_ctc(self, tokens):
        hidden_state = self.token_to_hidden_state(tokens)
        logits = self.lm_head(hidden_state)

        # CTCLoss doesn't support fp16
        log_probs = F.log_softmax(
            logits, dim=-1, dtype=torch.float32
        ).transpose(0, 1)  # [N, T, C] -> [T, N, C]
        return log_probs
    
    def get_ctc_loss(self, tokens, text_id):
        logits = self.token_to_ctc(tokens)
        # TODO: (QQ) fix loss function
        recon_feature, feature = torch.zeros((1)), torch.zeros((1))
        return self.criterion(recon_feature, feature, logits, text_id)

class BestRQVocoder(BestRQMelCTC):
    def __init__(
        self,
        frontend_config,
        encoder_config,
        codebook_config,
        vq_config=None,
        config=None,
    ):
        super().__init__(
            frontend_config=frontend_config,
            encoder_config=encoder_config,
            codebook_config=codebook_config,
            vq_config=vq_config,
        )
        del self.lm_head
        del self.unfolder
        del self.quantizer
        self.vocoder = BigVGAN(config)
        self.config = config

    def forward(self, input_dict):
        feature = input_dict["feature"]
        encoded_feature = self.frontend(feature)
        if self.vq is not None:
            model_output = self.encoder(encoded_feature, vq=self.vq)
            hidden_state = model_output["last_hidden_state"]
            quant_encoder_out, quant_idx, quant_loss = model_output["quant_state"]
        else:
            hidden_state = self.encoder(encoded_feature)["last_hidden_state"]
        recon_feature = self.spec_reconstructor(hidden_state)
        recon_wav = self.vocoder(hidden_state.transpose(1, 2)).squeeze(1)
        output_dict = {"recon_feature": recon_feature, "recon_wav": recon_wav}
        if self.vq is not None:
            output_dict.update(
                {
                    "vq_states": quant_encoder_out,
                    "vq_ids": quant_idx,
                    "vq_loss": quant_loss,
                }
            )
        return output_dict
