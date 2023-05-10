""" Base wav2vec2 model """

import torch
from torch import nn
from core.models.pretrained.frontend import ConvFeatureExtractionModel
from core.models.pretrained.quantizer import GumbelVectorQuantizer
from core.criterions import *
from core.models.pretrained.encoder import *
from core.models.pretrained.utils import *
from core.models.aed.classifier import *
from core.extensions import PipelineModule, LayerSpec


class PreProcess(nn.Module):
    '''pre process'''

    def __init__(self, args, padding_mask, mask, features_only, reduce):
        super().__init__()
        self.recv_grad_idx = (0, 1, 2, 3) if args.negatives_from_everywhere else (0, 1, 3)
        self.args = args
        self.mask = mask
        self.features_only = features_only
        self.padding_mask = padding_mask
        self.reduce = reduce
        self.feature_grad_mult = args.feature_grad_mult

        if self.args.get("wav2vec_use_fbank", False):
            self.embed = self.args.fbank_conv_embed_dim
            self.fbank_convs = nn.Sequential(
                nn.Conv1d(80, self.embed, 3, 2, 1),
                nn.GELU(),
                nn.Conv1d(self.embed, self.embed, 3, 2, 1),
                nn.GELU(),
            )
        else:
            feature_enc_layers = eval(args.conv_feature_layers)
            self.embed = feature_enc_layers[-1][0]
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=args.extractor_mode,
                conv_bias=args.conv_bias,
                conv_type=args.wav2vec_feature_conv_type,
            )

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.mask_minlen = args.mask_minlen_type
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.mask_channel_minlen = args.mask_channel_minlen_type
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space
        self.mask_emb = nn.Parameter(torch.FloatTensor(args.encoder_embed_dim).uniform_())

        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

        self.layer_norm = nn.LayerNorm(self.embed)
        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
        )

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.input_quantizer = None
        if args.quantize_input:
            if args.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=args.latent_vars,
                    temp=eval(args.latent_temp),
                    groups=args.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, args.encoder_embed_dim)

    def apply_mask(self, x, padding_mask):
        """apply_mask"""
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
                self.mask_minlen,
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
                self.mask_channel_minlen,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward(self, inputs):
        '''forward'''
        # pylint:disable=too-many-branches
        batch_data = dict()
        batch_data['waveform'] = inputs[0]
        batch_data['wav_mask'] = inputs[1]
        batch_data['src_mask'] = inputs[2]

        padding_mask = (1 - batch_data['src_mask']).int().bool()

        if self.args.wav2vec_use_fbank:
            fbank = batch_data['fbank']
            features = self.fbank_convs(fbank.transpose(1, 2).contiguous())
        else:
            wav = batch_data['waveform'].float()
            if self.args.normalize:
                with torch.no_grad():
                    wav = F.layer_norm(wav, wav.shape)
            else:
                wav.div_(32768)
            source = wav.type_as(batch_data['src_mask'])

            if self.feature_grad_mult > 0:
                features = self.feature_extractor(source)
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            features = self.project_inp(features)

        if self.mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        return (
            x,
            y,
            unmasked_features,
            features_pen,
            padding_mask,
            mask_indices,
            batch_data['src_mask'],
        )


class PosConv(nn.Module):
    '''pos conv'''

    def __init__(self, args):
        super().__init__()
        self.send_grad_idx = (
            (
                0,
                1,
                2,
                3,
            )
            if args.negatives_from_everywhere
            else (0, 1, 3)
        )
        self.recv_grad_idx = (
            (
                0,
                1,
                2,
                3,
            )
            if args.negatives_from_everywhere
            else (0, 1, 3)
        )
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

    def forward(self, inputs):
        '''forward'''
        x, y, unmasked_features, features_pen, padding_mask, mask_indices, src_mask = inputs

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2).contiguous())
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        return (x, y, unmasked_features, features_pen, padding_mask, mask_indices, src_mask)


class TransfoemerPipe(nn.Module):
    '''transformer pipe'''

    def __init__(self, args, idx):
        super().__init__()
        self.send_grad_idx = (0, 1, 2, 3) if args.negatives_from_everywhere else (0, 1, 3)
        self.recv_grad_idx = (0, 1, 2, 3) if args.negatives_from_everywhere else (0, 1, 3)

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.layerdrop = args.encoder_layerdrop
        self.fused_transformer = args.get('fused_transformer', True)
        self.idx = idx
        self.args = args

        self.layer = BiTransformerLayer(
            embed_dim=self.embedding_dim,
            attention_heads=args.encoder_attention_heads,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_dropout=args.attention_dropout,
            hidden_dropout=self.dropout,
            activation_dropout=args.activation_dropout,
            activation=args.activation_fn,
            normalize_before=args.layer_norm_first,
            squeeze_mem=args.get('squeeze_mem', True),
        )

        self.apply(self.reset_parameters)

    @staticmethod
    def reset_parameters(module):
        """
        Initialize the weights specific to the BERT Model.
        This overrides the default initializations depending on the specified arguments.
            1. If normal_init_linear_weights is set then weights of linear
            layer will be initialized using the normal distribution and
            bais will be set to the specified value.
            2. If normal_init_embed_weights is set then weights of embedding
            layer will be initialized using the normal distribution.
            3. If normal_init_proj_weights is set then weights of
            in_project_weight for MultiHeadAttention initialized using
            the normal distribution (to be validated).
        """

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            if module.merge_qkv:
                module.in_proj.weight.data.normal_(mean=0.0, std=0.02)
            else:
                module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
                module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
                module.v_proj.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inputs):
        '''forward'''
        x, y, unmasked_features, features_pen, padding_mask, mask_indices, src_mask = inputs
        dropout_probability = random.random() if self.training else 1
        if dropout_probability > self.layerdrop:
            x = self.layer(
                x, encoder_padding_mask=padding_mask, fused=self.fused_transformer, batch_first=True
            )
        return (x, y, unmasked_features, features_pen, padding_mask, mask_indices, src_mask)


class PostProcess(nn.Module):
    '''post process'''

    def __init__(self, args):
        super().__init__()
        self.send_grad_idx = (0, 1, 2, 3) if args.negatives_from_everywhere else (0, 1, 3)
        self.recv_grad_idx = (0, 1, 2, 3)

        self.quantizer = None
        self.args = args
        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

        if self.args.wav2vec_use_fbank:
            self.embed = self.args.fbank_conv_embed_dim
        else:
            feature_enc_layers = eval(args.conv_feature_layers)
            self.embed = feature_enc_layers[-1][0]

        if args.quantize_targets:
            vq_dim = args.latent_dim if args.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=args.latent_vars,
                temp=eval(args.latent_temp),
                groups=args.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        self.n_negatives = args.num_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere
        self.cross_sample_negatives = args.cross_sample_negatives

        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())

        self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)
        self.transformer_layernorm = nn.LayerNorm(args.encoder_embed_dim)

    def sample_negatives(self, y, num):
        """sample_negatives"""
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()
                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * num))
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(bsz, num, self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def forward(self, inputs):
        '''forward'''
        x, y, unmasked_features, features_pen, padding_mask, mask_indices, src_mask = inputs

        if self.args.layer_norm_first:
            self.transformer_layernorm(x)

        num_vars = torch.cuda.FloatTensor([0])
        code_ppl = torch.cuda.FloatTensor([0])
        prob_ppl = torch.cuda.FloatTensor([0])
        curr_temp = torch.cuda.FloatTensor([0])

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = torch.cuda.FloatTensor([q["num_vars"]])
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = torch.cuda.FloatTensor([q["temp"]])
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _negs_idx = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _negs_idx = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        return (
            x,
            y,
            features_pen,
            negs,
            padding_mask,
            src_mask,
            num_vars,
            code_ppl,
            prob_ppl,
            curr_temp,
        )


class PipelinePretrainedWav2Vec2(PipelineModule):
    """PipelinePretrainedWav2Vec2"""

    def __init__(self, args, padding_mask=None, mask=True, features_only=False, reduce=True):
        '''init'''
        self.args = args
        self.mask = mask
        self.features_only = features_only
        self.reduce = reduce
        self.logit_temp = args.logit_temp
        self.num_updates = 0
        self.data_dtype_key_map = {
            torch.float32: ["waveform", "wav_mask", "src_mask"],
            torch.float64: None,
            torch.int32: None,
            torch.int64: None,
        }

        layers = []
        layers.append(
            LayerSpec(
                PreProcess,
                args,
                padding_mask=padding_mask,
                mask=mask,
                features_only=features_only,
                reduce=reduce,
            )
        )
        layers.append(LayerSpec(PosConv, args))
        for idx in range(args.encoder_layers):
            layers.append(LayerSpec(TransfoemerPipe, args, idx))
        layers.append(LayerSpec(PostProcess, args))

        # pylint: disable=unexpected-keyword-arg
        super().__init__(layers=layers, loss_fn=self._loss_fn, cfg=self.args.pipeline)

        self.infonce = args.infonce
        self.loss_weights = eval(args.loss_weights)

    def _loss_fn(self, inputs, target):
        '''loss func'''
        (
            x,
            y,
            features_pen,
            negs,
            padding_mask,
            src_mask,
            num_vars,
            code_ppl,
            prob_ppl,
            curr_temp,
        ) = inputs
        num_vars = num_vars.item()
        curr_temp = curr_temp.item()
        batch_data = {'src_mask': src_mask}
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        logits = self.get_logits(result).float()
        target = self.get_targets(batch_data, result)

        weights = None
        if hasattr(self, 'get_target_weights') and not self.infonce:
            weights = self.get_target_weights(target, result)
            if torch.is_tensor(weights):
                weights = weights.float()

        forward_out = wav2vec_criterion(
            logits,
            target,
            result,
            batch_data,
            weights,
            self.reduce,
            self.infonce,
            self.loss_weights,
        )
        if self.training:
            self.num_updates += 1

        return forward_out

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.num_updates = num_updates
        self.encoder.num_updates = num_updates
        if self.quantizer:
            self.quantizer.set_num_updates(num_updates)

    @staticmethod
    def get_logits(net_output):
        """get_logits"""
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def compute_preds(self, x, y, negatives):
        """compute_preds"""
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    # pylint: disable=unused-argument
    @staticmethod
    def get_targets(sample, net_output, expand_steps=True):
        """get_targets"""
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)
