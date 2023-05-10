""" Base WavLM model """

import torch
from torch import nn
from core.models.layers.embedding import RelLearnedPositionalEmbedding
from core.models.pretrained.frontend import ConvFeatureExtractionModel
from core.models.pretrained.encoder import TransformerEncoder
from core.models.pretrained.utils import *
from core.utils import get_rank, get_world_size


class PretrainedWavLM(nn.Module):
    '''PretrainedWavLM'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        # We should use all the outputs of the intermediate layers
        args.setdefault('output_layer_result', True)
        self.args.distributed_rank = get_rank()
        self.args.distributed_world_size = get_world_size()

        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
            conv_type=args.get("wavlm_feature_conv_type", "default"),
        )

        self.post_extract_proj = nn.Linear(self.embed, args.encoder_embed_dim)

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)

        self.feature_grad_mult = args.feature_grad_mult

        self.mask_emb = nn.Parameter(torch.FloatTensor(args.encoder_embed_dim).uniform_())

        self.encoder = TransformerEncoder(args)
        self.layer_norm = nn.LayerNorm(self.embed)

        # relative positional embedding
        self.pos_enc = RelLearnedPositionalEmbedding(
            args.num_buckets, args.encoder_attention_heads, args.max_distance
        )
        nn.init.xavier_normal_(self.pos_enc.weight)

    def apply_mask(self, x, padding_mask):
        '''apply_mask'''
        # pylint: disable=invalid-name
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb.type_as(x)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward(self, batch_data, mask=True):
        '''forward
        Args:
            batch_data:
                waveform: the waveform (range from [-32768 ~ 32767])
                src_mask: the mask where the valid data is 1
            mask: whether to use random mask on input feature (like SpecAug)
        '''
        wav = batch_data['waveform'].float()
        if batch_data['src_mask'] is not None:
            padding_mask = (1 - batch_data['src_mask']).int().bool()
            valid_len = batch_data['src_mask'].sum(1, keepdim=True)
        else:
            padding_mask = None
            valid_len = torch.ones_like(wav).type_as(wav).sum(1, keepdim=True)

        # change in-place divide to the normal one, since the input may be used outside.
        wav = wav / 32768
        if self.args.normalize:
            with torch.no_grad():
                # To match the layer_norm in unispeech: wav = F.layer_norm(wav, wav.shape)
                # F.layer_norm does not support norm with mask
                mean = torch.sum(wav, dim=-1, keepdim=True) / valid_len
                std = torch.sqrt(
                    torch.sum((wav - mean) ** 2, dim=-1, keepdim=True) / valid_len + 1e-05
                )
                wav = (wav - mean) / std
                if batch_data['src_mask'] is not None:
                    wav = wav * batch_data['src_mask']
        source = wav

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        # B x C x T -> B x T x C
        features = features.transpose(1, 2)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        features = self.layer_norm(features)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)

        if mask:
            x, _ = self.apply_mask(features, padding_mask)
        else:
            x = features

        pos_emb = self.pos_enc(x)
        x, layer_results = self.encoder.forward(
            x, padding_mask, pos_emb=pos_emb, output_all_layers=True, include_first_layer=True
        )

        return {
            'x': x,
            'padding_mask': padding_mask,
            'features': features,
            'layer_results': layer_results,
        }
