""" Base wav2vec2 model """

import torch
from torch import nn
from core.models.pretrained.frontend import ConvFeatureExtractionModel
from core.models.pretrained.quantizer import GumbelVectorQuantizer
from core.criterions import *
from core.models.pretrained.encoder import *
from core.models.pretrained.utils import *
from core.models.aed.classifier import *
from core.utils import get_rank, get_world_size


class PretrainedWav2Vec2(nn.Module):
    """PretrainedWav2Vec2"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.distributed_rank = get_rank()
        self.args.distributed_world_size = get_world_size()
        self.num_updates = 0
        if self.args.get("wav2vec_use_fbank", False):
            self.embed = self.args.fbank_conv_embed_dim
            self.fbank_convs = nn.Sequential(
                nn.Conv1d(self.args.get("fbank_dim", 80), self.embed, 3, 2, 1),
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
        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
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

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = args.num_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere
        self.cross_sample_negatives = args.cross_sample_negatives

        self.logit_temp = args.logit_temp

        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

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

        self.mask_emb = nn.Parameter(torch.FloatTensor(args.encoder_embed_dim).uniform_())

        # TODO(zhengyijie): consider support conformer or dfsmn encoder
        self.encoder = TransformerEncoder(args)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())

        self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)

        self.infonce = args.infonce
        self.loss_weights = eval(args.loss_weights)

        if args.wav2vec_finetuning:
            self.remove_pretraining_modules()

        self._register_load_state_dict_pre_hook(self._model_load_hook)

    @staticmethod
    def convert_huggingface_chkpt(pretrained_state_dict):
        """convert_huggingface_chkpt"""
        res = dict()
        for name, val in pretrained_state_dict.items():
            new_name = name.replace('wav2vec2.', '')
            if new_name == "masked_spec_embed":
                new_name = "mask_emb"
            elif new_name.startswith('feature_extractor.conv_layers'):
                new_name = new_name.replace('.conv.', '.0.')
                new_name = new_name.replace('.layer_norm.', '.2.1.')
            elif new_name.startswith('encoder.layers'):
                new_name = new_name.replace('.attention.', '.self_attn.')
                new_name = new_name.replace('.layer_norm.', '.self_attn_layer_norm.')
                new_name = new_name.replace('.feed_forward.intermediate_dense.', '.fc1.')
                new_name = new_name.replace('.feed_forward.output_dense.', '.fc2.')
            elif new_name.startswith('encoder.pos_conv'):
                new_name = new_name.replace('.pos_conv_embed.conv.', '.pos_conv.0.')
            elif new_name.startswith('feature_projection.projection'):
                new_name = new_name.replace('feature_projection.projection', 'post_extract_proj')
            res[new_name] = val
        return res

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
        """Transform huggingface-chkpt to dolphin-chkpt"""
        if self.args.get('is_pipe_model', False):
            return
        old_state_dict = state_dict.copy()
        state_dict.clear()
        for name, param in old_state_dict.items():
            name = name.replace('wav2vec_model.', prefix)
            new_name = name.replace('wav2vec2.', prefix)
            if new_name == prefix + "masked_spec_embed":
                new_name = prefix + "mask_emb"
            elif new_name.startswith(prefix + 'feature_extractor.conv_layers'):
                new_name = new_name.replace('.conv.', '.0.')
                new_name = new_name.replace(
                    '.layer_norm.', '.2.1.' if self.args.extractor_mode == 'layer_norm' else '.2.'
                )
            elif new_name.startswith(prefix + 'encoder.layers'):
                new_name = new_name.replace('.attention.', '.self_attn.')
                new_name = new_name.replace('.layer_norm.', '.self_attn_layer_norm.')
                new_name = new_name.replace('.feed_forward.intermediate_dense.', '.fc1.')
                new_name = new_name.replace('.feed_forward.output_dense.', '.fc2.')
            elif new_name.startswith(prefix + 'encoder.pos_conv'):
                new_name = new_name.replace('.pos_conv_embed.conv.', '.pos_conv.0.')
            elif new_name.startswith(prefix + 'feature_projection'):
                new_name = new_name.replace('feature_projection.projection', 'post_extract_proj')
                new_name = new_name.replace('feature_projection.layer_norm', 'layer_norm')
            state_dict[new_name] = param

    def apply_mask(self, x, padding_mask, **kwargs):
        """apply_mask"""
        # pylint: disable=invalid-name
        B, T, C = x.shape
        mask_indices = kwargs.get('mask_idc', None)
        if self.mask_prob > 0:
            if mask_indices is None:
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
                mask_indices = torch.from_numpy(mask_indices)
            mask_indices = mask_indices.to(x.device)
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

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.num_updates = num_updates
        self.encoder.num_updates = num_updates
        if self.quantizer:
            self.quantizer.set_num_updates(num_updates)

    def forward(
        self,
        batch_data,
        padding_mask=None,
        mask=True,
        features_only=False,
        reduce=True,
        output_layer=None,
    ):
        """forward"""
        # pylint:disable=too-many-branches,too-many-locals,too-many-statements

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

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            mask_idc = batch_data.get('mask_idc', None)
            x, mask_indices = self.apply_mask(features, padding_mask, mask_idc=mask_idc)
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

        x, _ = self.encoder(
            x, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
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
            logits, target, result, batch_data, weights, reduce, self.infonce, self.loss_weights
        )
        if self.training:
            self.num_updates += 1

        return forward_out

    def forward_freeze_internal(
        self,
        batch_data,
        mask=True,
        freeze_layers=-1,
        output_feature_extractor=False,
        output_hidden_states=False,
        features_only=False,
    ):
        """forward_freeze_internal"""
        padding_mask = (1 - batch_data['src_mask']).int().bool()

        with torch.no_grad():
            if self.args.wav2vec_use_fbank:
                fbank = batch_data['fbank']
                features = self.fbank_convs(fbank.transpose(1, 2).contiguous())
            else:
                wav = batch_data['waveform'].float()
                if self.args.normalize:
                    wav_shape = [int(s) for s in wav.shape]
                    wav = F.layer_norm(wav, wav_shape)
                    # wav = F.layer_norm(wav)
                else:
                    wav.div_(32768)
                source = wav.type_as(batch_data['src_mask'])

                features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        extracted_features = features if output_feature_extractor or features_only else None

        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            padding_mask = padding_mask[:, : padding_mask.size(1) - extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask[:, :, -1]

        if features_only:
            return {
                'encoder_output': extracted_features,
                'padding_mask': padding_mask,
                'extracted_features': None,
                'hidden_states': None,
            }

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        x = features
        if mask:
            mask_idc = batch_data.get('mask_idc', None)
            x, _ = self.apply_mask(features, padding_mask, mask_idc=mask_idc)

        # internal freeze
        if self.args.get('freeze_encoder', 'True'):
            x, hidden_states = self.encoder.forward_freeze_internal(
                x,
                padding_mask=padding_mask,
                freeze_layers=freeze_layers,
                output_hidden_states=output_hidden_states,
            )
        else:
            x, hidden_states, _ = self.encoder.forward_attention(
                x,
                padding_mask=padding_mask,
                output_hidden_states=output_hidden_states,
            )

        return {
            'encoder_output': x,
            'padding_mask': padding_mask,
            'extracted_features': extracted_features,
            'hidden_states': hidden_states,
        }

    def quantize(self, x):
        """quantize"""
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, batch_data, mask=False):
        """extract_features"""
        res = self.forward(batch_data, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def inference(self, net_output):
        '''inference'''
        encoder_out = net_output["encoder_out"]  # T, B, C
        lprobs = F.log_softmax(encoder_out.float(), dim=-1).contiguous()

        non_padding_mask = ~net_output["padding_mask"]
        input_lengths = non_padding_mask.long().sum(-1).cpu()

        lprobs = lprobs.transpose(0, 1).contiguous().cpu()

        if self.args.w2l_decoder == 'wfst':
            return lprobs, input_lengths

        hyps = self.w2l_decoder.decode(lprobs, input_lengths)
        results = []
        for hyp in hyps:
            hyp = hyp[0]  # top1
            words = hyp.get('words')
            if words is None:
                tokens = self.args.tgt_dict.string(hyp['tokens'].tolist())
                words = self.hyp_post_process(tokens).split()
            results += [' '.join(words)]

        return results

    @staticmethod
    def get_logits(net_output):
        """get_logits"""
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    # pylint: disable=unused-argument
    @staticmethod
    def get_targets(sample, net_output, expand_steps=True):
        """get_targets"""
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def remove_pretraining_modules(self):
        """remove_pretraining_modules"""
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    @staticmethod
    def reorder_encoder_out(encoder_out, new_order):
        """reorder_encoder_out"""
        # encoder_out shape (B,T,N) new_order shape (B*beam)
        for name, tensor in encoder_out.items():
            encoder_out[name] = tensor.index_select(0, new_order)
        return encoder_out

    def forward_attention(
        self, batch_data, output_feature_extractor=None, output_hidden_states=None
    ):
        """forward_attention"""
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
            features = self.feature_extractor(source)

        if output_feature_extractor is not None:
            output_feature_extractor = features

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask[:, :, -1]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        outputs = self.encoder.forward_attention(features, padding_mask=padding_mask)

        # return outputs
        return {
            'encoder_output': outputs[0],
            'padding_mask': padding_mask,
            'output_feature_extractor': output_feature_extractor,
            'output_hidden_states': outputs[1],
        }
