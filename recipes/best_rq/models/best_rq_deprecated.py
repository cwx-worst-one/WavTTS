import torch
from torch import nn
from samantha.utils.hparams import DotDict
from core.models.pretrained.utils import compute_mask_indices
from core.models.asr.acoustic_frontend import Conv2dPooling
from core.models.asr.acoustic_backbone import MaskedConformerBackbone
from core.models.pretrained.quantizer import RandomProjectionQuantizer
from core.criterions.criterion import Xentropy


class BestRq(nn.Module):

    def __init__(
        self,
        # frontend
        n_mels=128,
        front_end_conv0_ch=256,
        front_end_conv1_ch=256,
        front_end_padding=1,
        downsampling_size=4,
        # backbone
        conformer_normalize_before=True,
        conformer_attention_heads=16,
        conformer_mask_topology=None,
        conformer_linear_units=1024,
        conformer_num_blocks=24,
        conformer_dropout_rate=0.1,
        conformer_positional_dropout_rate=0.1,
        conformer_attention_dropout_rate=0.1,
        conformer_positionwise_layer_type='linear',
        conformer_activation_fn='gelu',
        conformer_positionwise_conv_kernel_size=1,
        conformer_macaron_style=1,
        conformer_pos_enc_layer_type='fix_rel_pos',
        conformer_selfattention_layer_type='rel_selfattn',
        conformer_layer_order='mhsa_before_conv',
        conformer_use_cnn_module=1,
        conformer_cnn_module='ConvolutionModule',
        conformer_cnn_module_kernel='5',
        conformer_cnn_norm_type='layer_norm',
        conformer_layernorm_interval=0,
        conformer_weight_scale=1.0,
        conformer_half_pooling=0,
        backbone_memory_size=1024,
        dropout=0.1,
        # BEST-RQ
        n_softmax=8,
        mask_prob=0.3,
        mask_span=40,  # 0.4s
        mask_noise_std=0.1,
        codebook_size=4096,
        codebook_dim=16,
        # loss
        label_smooth_factor=0,
    ):
        super().__init__()
        quantizer_input_dim = n_mels * 9
        self.n_softmax = n_softmax
        self.codebook_size = codebook_size
        self.mask_prob = mask_prob
        self.mask_span = mask_span
        self.mask_noise_std = mask_noise_std
        self.unfolder = nn.Unfold(
            kernel_size=(3, 1),
            dilation=1,
            padding=(front_end_padding, 0),
            stride=(2, 1)
        )
        self.quantizer = RandomProjectionQuantizer(
            input_dim=quantizer_input_dim,
            codebook_dim=codebook_dim,
            codebook_size=codebook_size,
            quantizer_num=self.n_softmax
        )
        self.proj_heads = nn.Linear(
            backbone_memory_size, n_softmax * codebook_size, bias=False
        )
        self.acoustic_front_end_module = Conv2dPooling(
            DotDict(
                {
                    "fbank_dim": n_mels,
                    "front_end_conv0_ch": front_end_conv0_ch,
                    "front_end_conv1_ch": front_end_conv1_ch,
                    "front_end_padding": front_end_padding,
                    "downsampling_size": downsampling_size,
                    "backbone_memory_size": backbone_memory_size,
                }
            )
        )
        self.encoder_backbone = MaskedConformerBackbone(
            DotDict(
                {
                    "backbone_memory_size": backbone_memory_size,
                    "conformer_normalize_before": conformer_normalize_before,
                    "conformer_attention_heads": conformer_attention_heads,
                    "conformer_mask_topology": conformer_mask_topology,
                    "conformer_linear_units": conformer_linear_units,
                    "conformer_num_blocks": conformer_num_blocks,
                    "conformer_dropout_rate": conformer_dropout_rate,
                    "conformer_positional_dropout_rate": conformer_positional_dropout_rate,
                    "conformer_attention_dropout_rate": conformer_attention_dropout_rate,
                    "conformer_positionwise_layer_type": conformer_positionwise_layer_type,
                    "conformer_activation_fn": conformer_activation_fn,
                    "conformer_positionwise_conv_kernel_size": conformer_positionwise_conv_kernel_size,
                    "conformer_macaron_style": conformer_macaron_style,
                    "conformer_pos_enc_layer_type": conformer_pos_enc_layer_type,
                    "conformer_selfattention_layer_type": conformer_selfattention_layer_type,
                    "conformer_layer_order": conformer_layer_order,
                    "conformer_use_cnn_module": conformer_use_cnn_module,
                    "conformer_cnn_module": conformer_cnn_module,
                    "conformer_cnn_module_kernel": conformer_cnn_module_kernel,
                    "conformer_cnn_norm_type": conformer_cnn_norm_type,
                    "conformer_layernorm_interval": conformer_layernorm_interval,
                    "conformer_weight_scale": conformer_weight_scale,
                    "conformer_half_pooling": conformer_half_pooling,
                    "dropout": dropout,
                }
            )
        )
        self.proj_heads = nn.Linear(
            backbone_memory_size, n_softmax * codebook_size, bias=False
        )
        self.criterion = Xentropy(
            DotDict(
                {
                    "label_smooth_factor": label_smooth_factor,
                }
            )
        )
    
    def _unfold(self, feature, feature_mask):
        b, t, d = feature.size()
        unfold_feature = (
            self.unfolder(feature.unsqueeze(1))
            .reshape(b, 3, -1, d)
            .transpose(1, 2)
            .reshape(b, -1, 3 * d)
        )
        unfold_fbank_mask = (
            self.unfolder(feature_mask.unsqueeze(1).unsqueeze(3))
            .reshape(b, 3, -1, 1)
            .transpose(1, 2)
            .reshape(b, -1, 3)
            .sum(-1)
            > 0
        ).float()
        return unfold_feature, unfold_fbank_mask
    
    def _subsample(self, feature, feature_mask):
        feature, feature_mask = self._unfold(feature, feature_mask)
        feature, feature_mask = self._unfold(feature, feature_mask)
        return feature, feature_mask

    def _gen_mask_indicators(self, input_masks):
        """Generate mask indicators, where 1 means masked out."""
        B, T = input_masks.shape
        mask_indicators = compute_mask_indices(
            shape=(B, T),
            padding_mask=1 - input_masks,
            mask_prob=self.mask_prob,
            mask_length=self.mask_span,
            mask_type="static",
            mask_other=0.0,
            mask_minlen_type="crop_end",
            min_masks=1,
            no_overlap=False,
            min_space=0,
            require_same_masks=True,
            mask_dropout=0.0,
        )
        mask_indicators = (
            torch.from_numpy(mask_indicators).to(input_masks.device).long() * input_masks
        )
        return mask_indicators

    # @torch.no_grad()
    # def get_latent(self, x, layer_ix=12):
    #     x = self.conv(x)
    #     emb = self.w2v_conformer(x, output_hidden_states=True)["hidden_states"]
    #     return emb[layer_ix]

    def forward(self, batch):
        feature, feature_mask, codes, subsampled_mask_indicators = self.prepare_feature(batch)
        front_end_out, backbone_mask, frontend_shape = self.acoustic_front_end_module(feature, feature_mask)
        encoder_backbone_out = self.encoder_backbone(front_end_out, backbone_mask, frontend_shape)

        b, t_enc, d = encoder_backbone_out.size()
        logits = self.proj_heads(encoder_backbone_out).view(
            b, t_enc, self.n_softmax, self.codebook_size
        )
        # (B, T, n_softmax, codebook_size)
        logits = logits.transpose(1, 2).reshape(
            b * self.n_softmax, t_enc, self.codebook_size
        )
        targets = (
            codes.view(b, t_enc, self.n_softmax)
            .transpose(1, 2)
            .reshape(b * self.n_softmax, t_enc)
        )
        backbone_mask = (
            backbone_mask.view(b, 1, t_enc)
            .repeat(1, self.n_softmax, 1)
            .reshape(b * self.n_softmax, t_enc)
        )
        subsampled_mask_indicators = (
            subsampled_mask_indicators.view(b, 1, t_enc)
            .repeat(1, self.n_softmax, 1)
            .reshape(b * self.n_softmax, t_enc)
        )
        target_mask = subsampled_mask_indicators * backbone_mask

        forward_out = self.criterion(
            logits=logits, src_mask=backbone_mask, target=targets, target_mask=target_mask
        )
        
        num_uni_code = sum([len(targets.view(b, self.n_softmax, t_enc)[i, j].unique()) for i in range(b) for j in range(self.n_softmax)]) / b / self.n_softmax
        return {
            "logits": logits,
            "loss": forward_out["backward_loss"],
            "accu": forward_out["acc"],
            "num_uni_code": num_uni_code,
        }

    @torch.no_grad()
    def prepare_feature(self, batch):
        feature = batch["feature"][:, :, :-1].transpose(1, 2)
        b, t, d = feature.size()
        feature_mask = torch.ones((b, t), dtype=feature.dtype, device=feature.device)
        mask_indicators = self._gen_mask_indicators(feature_mask)
        subsampled_feature, subsampled_mask_indicators = self._subsample(feature, mask_indicators)

        noise = self.mask_noise_std * torch.randn_like(feature, device=feature.device)
        feature = feature * (1.0 - mask_indicators)[:, :, None] + noise * mask_indicators[:, :, None]

        b, t_sub, d_sub = subsampled_feature.size()
        codes = self.quantizer(subsampled_feature.view(b * t_sub, d_sub))
        return feature, feature_mask, codes, subsampled_mask_indicators


# if __name__ == "__main__":
#     device = torch.device("cuda")
#     batch = {
#         "feature": torch.ones((2, 128, 3001), device=device)
#     }
#     model = BestRq().to(device)
#     x = model(batch)
#     breakpoint()
#     pass