import math
import torch
import random
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from recipes.umm2.models.umm_melrof import UMM

class MaskedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, mask=None):
        logits = logits.contiguous().float()
        targets = targets.contiguous()

        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, 1)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        loss = -torch.gather(log_probs, dim=1, index=targets)

        if mask is None:
            return loss.mean()

        mask = mask.contiguous()
        loss = loss.view(*mask.size()) * mask
        loss = (loss / mask.sum()).sum()
        return loss

def accuracy(rq_masked_logits, rq_masked_target):
    return (rq_masked_logits.argmax(1) == rq_masked_target).float().mean()


class RandomProjectionQuantizer(nn.Module):
    """RandomProjectionQuantizer"""

    def __init__(self, config):
        """A quantizer based on random projection
        See: https://arxiv.org/pdf/2202.01855.pdf
        Args:
            dim: input dimension (channels)
            codebook_size: the number of code in the codebook
            codebook_dim: the dimension of the the code
            codebook_num: the number of quantizers.
                    See multi-softmax in https://arxiv.org/abs/2303.01037
            initialization_type: the initialization method of the projection matrix
        """
        super().__init__()
        self.input_dim = config.rq_input_dim
        self.codebook_size = config.rq_codebook_size
        self.codebook_dim = config.rq_codebook_dim
        self.codebook_num = config.rq_codebook_num

        self.register_buffer(
            "prototypes",
            torch.zeros(
                self.codebook_num,
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
                self.codebook_num * self.codebook_dim,
                requires_grad=False,
            ),
        )
        self._initialize()

    def _initialize(self):
        """Initialize the parameters."""
        nn.init.normal_(self.prototypes)
        F.normalize(self.prototypes, dim=-1, out=self.prototypes)

        fan_in = self.input_dim
        fan_out = self.codebook_dim
        gain = 1.0
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        with torch.no_grad():
            self.proj.normal_(0, std)

    def forward(self, x):
        """
        Forward a batch of representations to get discrete codes.
        Args:
            x: [batch_size, dim]
        Returns:
            codes: [batch_size, codebook_num], torch.int64
        """
        assert len(x.shape) == 2
        batch_size, _ = x.shape

        projected = torch.matmul(
            x, self.proj
        )  # [batch_size, codebook_num*codebook_dim]
        projected = F.normalize(
            projected.view(batch_size, self.codebook_num, self.codebook_dim),
            p=2,
            dim=-1,
        )  # [batch_size, codebook_num, codebook_dim]
        projected = projected.permute(1, 0, 2).view(
            self.codebook_num, batch_size, 1, self.codebook_dim
        )  # [codebook_num, batch_size, 1, codebook_dim]

        # self.prototypes: [codebook_num, 1, codebook_size, codebook_dim]
        # it is normalized in the function _initialize

        # TODO: configure multiple distances, such as cosine similarity
        # distances = torch.norm(projected - self.prototypes, p=2, dim=-1)
        # [codebook_num, batch_size, codebook_size]

        # Save spaces.
        distances = (
            projected.view(self.codebook_num, batch_size, self.codebook_dim)
            .pow(2)
            .sum(-1, keepdim=True)  # [codebook_num, batch_size, 1]
            + self.prototypes.view(
                self.codebook_num, self.codebook_size, self.codebook_dim
            )
            .pow(2)
            .sum(-1, keepdim=True)
            .transpose(1, 2)  # [codebook_num, 1, codebook_size]
            - 2
            * torch.bmm(
                projected.view(self.codebook_num, batch_size, self.codebook_dim),
                self.prototypes.view(
                    self.codebook_num, self.codebook_size, self.codebook_dim
                ).transpose(1, 2),
            )  # [codebook_num, batch_size, codebook_size]
        )

        codes = torch.argmin(distances, dim=-1).transpose(0, 1)
        # [batch_size, codebook_num]
        return codes


class Masked_RQ(UMM):
    def __init__(
        self, 
        config,
        takes=["audio"],
        provides=["mel", "rq_logits", "rq_target", "flops", "loss", "accu"],
        bypasses=[],
    ):
        UMM.__init__(self, config, takes, provides, bypasses)

        self.config = config
        if config.rq_input_layernorm:
            self.rq_input_layernorm = nn.LayerNorm(
                config.n_mels
                * pow(config.feature_encoder_kernel, config.feature_encoder_padding),
                elementwise_affine=False,
            )
        self.unfolder = nn.Unfold(
            kernel_size=(config.feature_encoder_kernel, 1),
            dilation=1,
            padding=(config.feature_encoder_padding, 0),
            stride=(2, 1),
        )
        self.rq = RandomProjectionQuantizer(config)
        self.rq_head = nn.Linear(
            config.hidden_size,
            config.rq_codebook_size * config.rq_codebook_num,
            bias=False,
        )
        self.criterion = MaskedCrossEntropy()

    @torch.no_grad()
    def masking(self, x):
        mx = x.clone()
        b, t = mx.shape
        device = x.device

        # get random mask indices
        start_indices = (
            torch.rand(b, t // self.config.len_masking_raw, device=device)
            < self.config.mask_prob
        )
        if torch.all(start_indices == False):
            start_indices[
                random.randint(0, start_indices.size(0) - 1),
                random.randint(0, start_indices.size(1) - 1),
            ] = True
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.config.len_masking_raw, dim=1)
        )
        mel_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(
                self.config.len_masking_token * 4, dim=1
            )
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(self.config.len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(len(time_domain_masked_indices), dtype=x.dtype, device=device)
            * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise
        return mx, token_domain_masked_indices, mel_domain_masked_indices

    def get_masked_wav_to_mel(self, batch):
        wav = batch['audio'].squeeze(dim=1).float()
        wav = self.pad_audio(wav)
        mel = self.preprocessing(wav)["mel"]
        masked_audio, masked_indices, _ = self.masking(wav)
        return {"mel": mel, "masked_audio": masked_audio, "masked_indices": masked_indices}
        
    def _unfold(self, feature):
        d = feature.size(-1)
        unfold_feature = self.unfolder(feature.unsqueeze(1))
        unfold_feature = rearrange(unfold_feature, "b c (t d) -> b t (c d)", d=d)
        return unfold_feature

    def _subsample(self, feature):
        feature = self._unfold(feature)
        feature = self._unfold(feature)
        return feature

    @torch.no_grad()
    def get_rq_target(self, feature):
        rq_input = rearrange(self._subsample(feature), "b t d -> (b t) d")
        if self.config.rq_input_layernorm:
            rq_input = self.rq_input_layernorm(rq_input)
        target_tokens = self.rq(rq_input)
        target_tokens = rearrange(target_tokens, "(b t) c -> b t c", b=feature.size(0))
        return target_tokens

    def get_metrics(self, masked_logits, masked_target):
        metric_dict = {}
        metric_dict["loss"] = self.criterion(masked_logits, masked_target)
        metric_dict["accu"] = accuracy(masked_logits, masked_target)
        return metric_dict

    @torch.no_grad()
    def get_spec(self, batch):
        input_dict = self.get_masked_wav_to_mel(batch)
        mel = input_dict["mel"]
        maksed_mel = input_dict["masked_mel"]
        return {
            "Original Mel": mel.transpose(1, 2),
            "Masked Mel": maksed_mel.transpose(1, 2),
        }

    def _compute(self, batch):
        input_dict = self.get_masked_wav_to_mel(batch)

        masked_audio = input_dict["masked_audio"]
        masked_indices = input_dict["masked_indices"]

        hidden_states = self.get_latent(masked_audio)
        logits = self.rq_head(hidden_states)
        logits = rearrange(
            logits, "b t (d c) -> b t d c", c=self.config.rq_codebook_num
        )
        masked_logits = logits[tuple(masked_indices.t())]
        masked_logits = rearrange(
            masked_logits, "b d c -> (b c) d", c=self.config.rq_codebook_num
        )

        feature = input_dict["mel"]
        target = self.get_rq_target(feature)
        masked_target = target[tuple(masked_indices.t())]
        masked_target = rearrange(masked_target, "b c -> (b c)")
        output_dict = {
            "mel": input_dict["mel"],
            "rq_logits": logits,
            "rq_target": target,
            "flops": 0,
        }
        metric_dict = self.get_metrics(masked_logits, masked_target)
        output_dict.update(metric_dict)  # include loss and flops

        return output_dict


if __name__ == "__main__":
    from recipes.umm2.models.config import UMMConfig
    config = UMMConfig(
        use_power_stft=True,
        mel_bands=16,
        num_feature=128,
        use_checkpoint=True,
        use_flash_attn=True,
        enforce_dropout=0.1,
        out_num_feature=64,
        use_avg_pool=False,
    )

    dummy_input = {'audio': torch.randn(2, 1, 24000 * 30).to("cuda") }

    model = Masked_RQ(config)
    model = model.to("cuda")
    
    output_dict = model(dummy_input)
    #print(output_dict.keys())
    print(output_dict['rq_logits'].shape, output_dict['rq_target'].shape, output_dict['mel'].shape, output_dict['loss'])
    