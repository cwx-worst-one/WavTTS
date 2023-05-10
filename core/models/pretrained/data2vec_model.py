""" Base data2vec model """

import torch
from torch import nn
from core.models.pretrained.frontend import ConvFeatureExtractionModel
from core.models.layers.ema_module import EMAModule
from core.criterions import *
from core.models.pretrained.encoder import *
from core.models.pretrained.utils import *
from core.models.aed.classifier import *
from core.utils import get_rank, get_world_size


class PretrainedData2Vec(nn.Module):
    '''PretrainedData2Vec'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.distributed_rank = get_rank()
        self.args.distributed_world_size = get_world_size()

        feature_enc_layers = eval(args.conv_feature_layers)
        self.extractor_embed = feature_enc_layers[-1][0]

        self.ema = None
        self.embed = args.encoder_embed_dim

        self.average_top_k_layers = args.average_top_k_layers
        self.loss_beta = args.loss_beta
        self.loss_scale = args.loss_scale

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
            conv_type=args.data2vec_feature_conv_type,
            recompute=args.get('conv_feature_recompute', False),
        )

        self.post_extract_proj = nn.Linear(self.extractor_embed, args.encoder_embed_dim)

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.mask_minlen = args.mask_minlen_type
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_before = args.mask_channel_before
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.mask_channel_minlen = args.mask_channel_minlen_type
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)

        self.feature_grad_mult = args.feature_grad_mult

        self.mask_emb = nn.Parameter(torch.FloatTensor(args.encoder_embed_dim).uniform_())

        self.encoder = TransformerEncoder(args)
        self.layer_norm = nn.LayerNorm(self.extractor_embed)

        self.final_proj = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)

        self.num_updates = 0

        if self.args.weighted_data2vec:
            init_value = [1.0 / self.average_top_k_layers] * self.average_top_k_layers
            self.weighted_coef = torch.nn.Parameter(torch.Tensor(init_value))

        self.cuda()

        if args.get('use_pretrained_ema', False):
            self.make_ema_teacher()

        if args.data2vec_finetuning:
            self.remove_pretraining_modules()

        if not args.get('lazy_init_ema', True):
            self.make_ema_teacher()

    def make_ema_teacher(self):
        '''make_ema_teacher'''
        skip_keys = set()
        if self.args.ema_layers_only:
            self.args.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")

        self.ema = EMAModule(
            self.encoder if self.args.ema_transformer_only else self,
            ema_decay=self.args.ema_decay,
            ema_fp32=True,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        '''set_num_updates'''
        self.num_updates = num_updates
        self.encoder.num_updates = num_updates

        if self.args.get('data2vec_without_ema', False):
            return

        if self.ema is None and self.final_proj is not None:
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.args.ema_decay != self.args.ema_end_decay:
                if num_updates >= self.args.ema_anneal_end_step:
                    decay = self.args.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.args.ema_decay,
                        self.args.ema_end_decay,
                        num_updates,
                        self.args.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.args.ema_transformer_only else self)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        # pylint: disable=unsupported-assignment-operation
        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        '''apply_mask'''
        # pylint: disable=invalid-name
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
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

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    mask_minlen_type=self.mask_minlen,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.args.require_same_masks,
                    mask_dropout=self.args.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb.type_as(x)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
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
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_ema(
        self,
        pre_encoder_features,
        padding_mask,
        orig_padding_mask,
        batch_data,
        mask_indices,
    ):
        '''forward EMA'''
        # pylint:disable=too-many-branches
        with torch.no_grad():
            self.ema.model.eval()

            if self.args.ema_transformer_only:
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features,
                    padding_mask=padding_mask,
                    min_layer=self.args.encoder_layers - self.average_top_k_layers,
                    eval_fused=self.training,
                )
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                y, padding_mask = self.ema.model.extract_features(
                    batch_data=batch_data,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            # B x T x C -> T x B x C
            target_layer_results = [l[2].transpose(0, 1) for l in y["layer_results"]]

            permuted = False
            if self.args.instance_norm_target_layer or self.args.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.args.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(tl.float(), running_mean=None, running_var=None, training=True)
                    for tl in target_layer_results
                ]

            if self.args.instance_norm_target_layer:
                target_layer_results = [F.instance_norm(tl.float()) for tl in target_layer_results]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.args.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:]) for tl in target_layer_results
                ]

            if self.args.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:]) for tl in target_layer_results
                ]

            if not self.args.weighted_data2vec:  # simply average target_layer_results
                y = sum(target_layer_results) / len(target_layer_results)

                if self.args.layer_norm_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

                if self.args.instance_norm_targets:
                    y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

                if not permuted:
                    y = y.transpose(0, 1)

                y = y[mask_indices]

        coe = None
        if self.args.weighted_data2vec:  # self.weighted_coef requires grad, weighted sum
            coe = torch.nn.functional.softmax(self.weighted_coef, dim=-1).view(
                1, 1, 1, self.average_top_k_layers
            )
            layer_res_flat = torch.stack(target_layer_results, dim=-1)  # BTCk or TBCk
            y = (coe * layer_res_flat).sum(dim=-1)
            if self.args.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.args.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices]

        return y, coe

    def forward(
        self,
        batch_data,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        _padding_count=None,
    ):
        '''forward'''
        # pylint:disable=too-many-branches
        padding_mask = (1 - batch_data['src_mask']).int().bool()
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

        features_pen = features.float().pow(2).mean()  # debug

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        pre_encoder_features = None
        if self.args.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            )
        else:
            x = features
            mask_indices = None

        if not features_only:
            y, coe = self.forward_ema(
                pre_encoder_features,
                padding_mask,
                orig_padding_mask,
                batch_data,
                mask_indices,
            )
            self.encoder.output_layer_result = False
        else:
            y = coe = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            # B x T x C -> T x B x C
            layer_results = [
                (l[0].transpose(0, 1), l[1], l[2].transpose(0, 1)) for l in layer_results
            ]
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        x = x[mask_indices]
        x = self.final_proj(x)

        result = {}
        forward_out = data2vec_criterion(
            x,
            y,
            self.loss_beta,
            self.loss_scale,
            result,
            self.training,
            self.num_updates,
            self.args.min_target_var,
            self.args.min_pred_var,
            self.ema,
            features_pen,
            batch_data,
            coe if self.args.weighted_data2vec else None,
        )

        if self.training:
            self.num_updates += 1

        return forward_out

    def extract_features(self, batch_data, padding_mask=None, mask=False, layer=None):
        '''extract_features'''
        res = self.forward(
            batch_data,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res["x"], res["padding_mask"]

    def remove_pretraining_modules(self, last_layer=None):
        '''remove_pretraining_modules'''
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class SharedEncoder(nn.Module):
    '''shared speech encoder'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d2v_model = eval(args.data2vec_type)(args)
        self.shared_layer_index = args.get('shared_layer_index', -1)
        self.norm_by_last_ln = args.get('shared_feature_normalized_lastln')
        self.freeze_feature_extractor = args.get('freeze_feature_extractor', False)
        self.mask_prob_text = args.get('mask_prob_text', args.mask_prob)
        self.mask_channel_text = args.get('mask_channel_text', args.mask_channel_prob)
        if args.get('remove_speech_encoder', False):
            self.freeze_feature_extractor = False
            self.d2v_model.ema = None
            self.d2v_model.final_proj = None
            self.d2v_model.layer_norm = None
            self.d2v_model.post_extract_proj = None
            self.d2v_model.feature_extractor = None
            if self.mask_prob_text <= 0:
                self.d2v_model.mask_emb = None
            if self.shared_layer_index >= 0:
                # self.d2v_model.encoder.pos_conv = None
                self.d2v_model.encoder.layers = self.d2v_model.encoder.layers[
                    self.shared_layer_index + 1 :
                ]
                self.shared_layer_index = -1
        self._register_load_state_dict_pre_hook(self._model_load_hook)
        self.train()

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.num_updates = num_updates
        self.d2v_model.set_num_updates(num_updates)

    def set_ema(self, ema_mod):
        '''set ema for data2vec'''
        # pylint:disable=missing-docstring
        class Proxy:
            def __init__(self, _ema):
                self.model = _ema

        self.d2v_model.ema = Proxy(ema_mod.d2v_model.encoder)

    def del_ema(self):
        '''del ema for data2vec'''
        self.d2v_model.ema = None

    # pylint:disable=invalid-name
    def apply_mask(self, x, mask, mask_prob, mask_channel_prob):
        '''apply mask'''
        B, T, C = x.shape
        if mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                mask,
                mask_prob,
                10,
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            noise = torch.randn_like(x) * 0.1
            x[mask_indices] = noise[mask_indices]

        if mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                mask_channel_prob,
                64,
                require_same_masks=False,
            )
            mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device)
            mask_channel_indices = mask_channel_indices.unsqueeze(1).expand(-1, T, -1)
            noise = torch.randn_like(x) * 0.1
            x[mask_channel_indices] = noise[mask_channel_indices]

        return x

    def forward_pretrained(self, batch_data):
        '''speech-only forward'''
        self.d2v_model.mask_channel_prob = 0
        forward_out = self.d2v_model(batch_data)
        self.d2v_model.mask_channel_prob = self.args.mask_channel_prob
        return forward_out

    def forward_text(self, ys, ys_mask, mask=True):
        '''text-only forward'''

        padding_mask = ys_mask
        if mask:
            x = self.apply_mask(
                ys.clone(), padding_mask, self.mask_prob_text, self.mask_channel_text
            )
        else:
            x = ys

        x, _ = self.d2v_model.encoder.forward_internal(
            x,
            padding_mask=padding_mask,
            layer=self.shared_layer_index,
        )

        return x, padding_mask

    def extract_features(self, batch_data, mask=False):
        '''extract features from shared encoder'''
        layer = self.shared_layer_index
        x, mask = self.d2v_model.extract_features(batch_data, mask=mask, layer=layer)
        if self.norm_by_last_ln:
            x = self.d2v_model.encoder.layer_norm(x)
        return x, mask

    def forward(self, batch_data):
        '''forward'''
        x, mask = self.d2v_model.extract_features(batch_data, mask=self.training)
        return x, mask

    def _model_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        """init from pretrained model"""
        old_state_dict = state_dict.copy()
        state_dict.clear()
        for name, param in old_state_dict.items():
            new_name = name
            if name.startswith('data2vec_ctc_model.'):
                # finetuned model
                new_name = name.replace('data2vec_ctc_model.', prefix)
            elif name.startswith('data2vec_model.'):
                # pretrained model
                new_name = name.replace('data2vec_model.', prefix + 'd2v_model.')
            state_dict[new_name] = param

    def train(self, mode: bool = True):
        '''
        Param will really be freezed by set param.requires_grad_(False).
        `with no_grad` may case param be updated by optimizer' weight_decay.
        What's more, we need to set module.eval() to fix some module buffer,
        such as BatchNorm.running_mean.
        '''
        super().train(mode)

        if self.freeze_feature_extractor:
            self.d2v_model.feature_extractor.requires_grad_(False)
            self.d2v_model.feature_extractor.train(False)
