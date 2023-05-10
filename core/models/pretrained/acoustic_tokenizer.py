""" acoustic tokenizers """

import os.path as osp
import copy
import logging
import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from core.models.pretrained.quantizer import RandomProjectionQuantizer
from core.models.asr.acoustic_frontend import *
from core.models.asr.acoustic_backbone import *
from core.models.codecs.soundstream import SoundStream
from core.utils.config import ConfigDict
from core.utils import dist_hdfs_get, dist_allreduce, ReduceOp, get_world_size
from core.extensions import AmpEnable

from packaging import version

is_torch_less_than_1_8 = version.parse(
    version.parse(torch.__version__).base_version
) < version.parse("1.8.0")


class RandomProjectionTokenizer(nn.Module):
    """RandomProjectionTokenizer
    a wrapper of RandomProjectionQuantizer
    """

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.quantizer_input_dim = args.acoustic_tokenizer_input_dim

        self.quantizer = RandomProjectionQuantizer(
            input_dim=self.quantizer_input_dim,
            codebook_size=self.args.acoustic_tokenizer_codebook_size,
            codebook_dim=self.args.acoustic_tokenizer_codebook_dim,
            prototype_initialization_method=self.args.prototype_initialization_method,
            projection_initialization_method=self.args.projection_initialization_method,
            quantizer_num=1,
        )

    def forward(
        self,
        x,
        x_masks,
    ):
        """forward"""
        batch_size, seq_len, dim = x.shape
        codes = self.quantizer(x.view(batch_size * seq_len, dim))
        codes = codes.reshape(batch_size, seq_len)
        return codes, x_masks


class KmeansTokenizer(nn.Module):
    """KmeansTokenizer"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.args.kmeans_model_path = getattr(args, "kmeans_model_path", "")
        self.args.kmeans_mean_and_var_stats_path = getattr(
            args, "kmeans_mean_and_var_stats_path", ""
        )
        self.args.return_distance = getattr(args, "return_distance", False)

        if self.args.kmeans_model_path:
            km_model, stats = self._load_kmeans_model(
                self.args.kmeans_model_path, self.args.kmeans_mean_and_var_stats_path
            )

            self.codebook_size = km_model.n_clusters
            self.feature_dim = km_model.cluster_centers_.shape[1]
            self.register_buffer("prototypes", torch.from_numpy(km_model.cluster_centers_))
            # [codebook_size, feature_dim]
        else:
            self.codebook_size = self.args.codebook_size
            self.feature_dim = self.args.feature_dim
            prototypes = torch.randn(self.codebook_size, self.feature_dim)
            logging.info("Initialize K-Means prototypes with Gaussian distributions.")
            self.register_buffer("prototypes", prototypes)
            stats = None

        if stats is not None:
            self.register_buffer("mean", torch.from_numpy(stats[0]))
            self.register_buffer("std", torch.from_numpy(stats[1]))
        else:
            self.mean = None
            self.std = None

        self.decay = getattr(args, "decay", 0.99)
        self.eps = getattr(args, "eps", 1e-5)
        self.register_buffer("cluster_size", torch.zeros(self.prototypes.shape[0]))
        self.register_buffer("prototypes_avg", self.prototypes.clone())

    def _load_kmeans_model(self, kmeans_model_path, kmeans_mean_and_var_stats_path):
        logging.info("Loading K-Means model from {}".format(kmeans_model_path))
        if kmeans_model_path.startswith("hdfs://"):
            local_dir = '/tmp/'
            tmp_model = 'kmeans_model.bin'
            dist_hdfs_get(kmeans_model_path, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            kmeans_model_path = tmp_model
        kmeans_model = joblib.load(kmeans_model_path)

        if kmeans_mean_and_var_stats_path:
            logging.info("Loading stats model from {}".format(kmeans_mean_and_var_stats_path))
            if kmeans_mean_and_var_stats_path.startswith("hdfs://"):
                tmp_stats = 'stats.npy'
                dist_hdfs_get(kmeans_mean_and_var_stats_path, local_dir, tmp_stats)
                kmeans_mean_and_var_stats_path = osp.join(local_dir, tmp_stats)
            stats = np.load(kmeans_mean_and_var_stats_path)
        else:
            stats = None
        return kmeans_model, stats

    @torch.no_grad()
    def _ema_update_clusters(self, codes, inputs, input_masks):
        """
        Args:
            codes: [batch_size]
            inputs: [batch_size, feature_dim]
            input_masks: [batch_size]
        """
        with AmpEnable(enabled=False):
            self.cluster_size = self.cluster_size.float()
            self.prototypes_avg = self.prototypes_avg.float()
            self.prototypes = self.prototypes.float()

            onehots = (
                F.one_hot(codes, self.codebook_size).type(self.prototypes.dtype)
                * input_masks[:, None].float()
            )  # [batch_size, codebook_size]

            embed_sum = torch.matmul(
                onehots.transpose(0, 1).float(), inputs.float()
            )  # [codebook_size, feature_dim]
            onehots_sum = torch.sum(onehots, dim=0).float()  # [codebook_size]
            if get_world_size() > 1:
                dist_allreduce(embed_sum, 'embed_sum', ReduceOp.SUM)
                dist_allreduce(onehots_sum, 'onehots_sum', ReduceOp.SUM)

            self.cluster_size.data.mul_(self.decay).add_(onehots_sum, alpha=1 - self.decay)
            self.prototypes_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )  # [codebook_size, feature_dim]
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            )  # [codebook_size]
            prototypes_avg_normalized = self.prototypes_avg / cluster_size.unsqueeze(1)
            self.prototypes.data.copy_(prototypes_avg_normalized)

    @torch.no_grad()
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [batch_size, feature_dim]
            input_masks: [batch_size] 1 for valid values, 0 for paddings
        Return:
            codes: [batch_size]
            input_masks: [batch_size]
        """

        if self.mean is not None:
            inputs = inputs - self.mean
        if self.std is not None:
            inputs = inputs / self.std

        distances = (
            inputs.pow(2).sum(1, keepdim=True)
            + self.prototypes.pow(2).sum(1, keepdim=True).transpose(0, 1)
            - 2 * torch.matmul(inputs, self.prototypes.transpose(0, 1))
        )
        # [batch_size, codebook_size]
        codes = torch.argmin(distances, dim=-1)

        if self.training:
            self._ema_update_clusters(codes, inputs, input_masks)

        if self.args.return_distance:  # for training k-means models
            with AmpEnable(enabled=False):
                clusters = F.embedding(codes, self.prototypes)
                dists = (inputs.float() - clusters.float()).pow(2).sum(-1).pow(
                    0.5
                ) * input_masks.float()
            return dists, input_masks

        return codes, input_masks


class KmeansTokenizerWithEncoder(KmeansTokenizer):
    """KmeansTokenizerWithEncoder"""

    def __init__(self, args):
        acoustic_tokenizer_args = self._get_acoustic_tokenizer_args(args)
        super().__init__(acoustic_tokenizer_args)

        self.acoustic_tokenizer_args = acoustic_tokenizer_args
        self.acoustic_front_end_module = eval(acoustic_tokenizer_args.front_end_type)(
            self.acoustic_tokenizer_args
        )
        self.acoustic_backbone_module = eval(acoustic_tokenizer_args.acoustic_backbone_type)(
            self.acoustic_tokenizer_args
        )

        args.acoustic_tokenizer_pretrained_model = getattr(
            args, 'acoustic_tokenizer_pretrained_model', ""
        )
        if args.acoustic_tokenizer_pretrained_model:
            self._load_pretrained_acoustic_tokenizer(args.acoustic_tokenizer_pretrained_model)

    def _load_pretrained_acoustic_tokenizer(self, acoustic_tokenizer_pretrained_model):
        if acoustic_tokenizer_pretrained_model.startswith("hdfs://"):
            local_dir = '/tmp/'
            tmp_model = 'pretrain_encoder_model.pt'
            dist_hdfs_get(acoustic_tokenizer_pretrained_model, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            pretrain_encoder_model_path = tmp_model
        else:
            pretrain_encoder_model_path = acoustic_tokenizer_pretrained_model
        pretrain_encoder_model = torch.load(pretrain_encoder_model_path, map_location='cpu')
        pretrain_encoder_model = pretrain_encoder_model['model']
        logging.info("Loaded pretrained tokenizer from {}".format(pretrain_encoder_model_path))
        msg = self.load_state_dict(pretrain_encoder_model, strict=False)
        err_msg = []
        if msg.unexpected_keys:
            err_msg.append(
                f'unexpected key in source state_dict: {", ".join(msg.unexpected_keys)}\n'
            )
        if msg.missing_keys:
            err_msg.append(f'missing keys in source state_dict: {", ".join(msg.missing_keys)}\n')
        if err_msg:
            err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
            logging.warning('\n'.join(err_msg))

    def _get_acoustic_tokenizer_args(self, args):
        internal_args = copy.deepcopy(args)
        acoustic_tokenizer_config = ConfigDict()
        for k, v in internal_args.items():
            if k.startswith("acoustic_tokenizer_"):
                new_k = k.replace("acoustic_tokenizer_", "")
                logging.info("Modify arg {} to {}".format(k, new_k))
                acoustic_tokenizer_config[new_k] = v
        return acoustic_tokenizer_config

    def frontend(self, fbank, mask):
        '''frontend'''
        if self.acoustic_front_end_module is not None:
            return self.acoustic_front_end_module(fbank, mask)
        # Nothing to do in frontend
        return fbank, mask, "BTN"

    def encoder_backbone(self, inputs, mask=None, frontend_shape="BTN"):
        '''Backbone'''
        attn_mask = False
        if (
            self.acoustic_tokenizer_args.acoustic_backbone_type
            in ('OfflineTransformerBackbone', 'RelTransformerBackbone', 'EmformerBackbone')
            and self.acoustic_backbone_module.causal_transformer
        ):
            attn_mask = True
        backbone_out = self.acoustic_backbone_module(
            inputs, mask, attn_mask=attn_mask, frontend_shape=frontend_shape
        )
        return backbone_out

    def forward(self, inputs, input_masks):
        with torch.no_grad():
            is_training = self.training
            # freeze encoder and bn
            self.eval()
            front_end_out, backbone_mask, frontend_shape = self.frontend(inputs, input_masks)
            encoder_backbone_out = self.encoder_backbone(
                front_end_out, backbone_mask, frontend_shape
            )

            batch_size, seq_len_enc, enc_dim = encoder_backbone_out.shape
            encoder_backbone_out = encoder_backbone_out.view(batch_size * seq_len_enc, enc_dim)

            if is_training:
                self.train()
            if self.args.return_distance:
                return super().forward(encoder_backbone_out, backbone_mask.view(-1))

            codes, masks = super().forward(encoder_backbone_out, backbone_mask.view(-1))
            # [batch_size, codebook_size]
            codes = codes.reshape(batch_size, seq_len_enc)
            masks = masks.reshape(batch_size, seq_len_enc)
            return codes, masks


class OriginalHubertTokenizer(nn.Module):
    """OriginalHubertTokenizer"""

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.args.acoustic_tokenizer_hubert_model_path = getattr(
            args, "acoustic_tokenizer_hubert_model_path", ""
        )
        self.args.acoustic_tokenizer_kmeans_model_path = getattr(
            args, "acoustic_tokenizer_kmeans_model_path", ""
        )
        self.args.acoustic_tokenizer_kmlayer = getattr(args, "acoustic_tokenizer_kmlayer", 9)

        from transformers import HubertModel

        self.model = HubertModel.from_pretrained(self.args.acoustic_tokenizer_hubert_model_path)
        self.model.eval()

        km_model = self._load_kmeans_model(self.args.acoustic_tokenizer_kmeans_model_path)

        self.codebook_size = km_model.n_clusters
        self.register_buffer("prototypes", torch.from_numpy(km_model.cluster_centers_))
        # [codebook_size, feature_dim]

    def _load_kmeans_model(self, kmeans_model_path):
        logging.info("Loading K-Means model from {}".format(kmeans_model_path))
        if kmeans_model_path.startswith("hdfs://"):
            local_dir = '/tmp/'
            tmp_model = 'kmeans_model.bin'
            dist_hdfs_get(kmeans_model_path, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            kmeans_model_path = tmp_model
        kmeans_model = joblib.load(kmeans_model_path)
        return kmeans_model

    def _torch_int_div(self, tensor1, tensor2):
        """
        A function that performs integer division across different versions of PyTorch.
        """
        if is_torch_less_than_1_8:
            return tensor1 // tensor2
        else:
            return torch.div(tensor1, tensor2, rounding_mode="floor")

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return self._torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(
            self.model.config.conv_kernel, self.model.config.conv_stride
        ):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(
            torch.long
        )
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                torch.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def forward(self, inputs, input_masks):
        with torch.no_grad():
            self.eval()
            outputs = self.model(inputs, attention_mask=input_masks, output_hidden_states=True)

            encoder_backbone_out = outputs.hidden_states[
                self.args.acoustic_tokenizer_kmlayer
            ].detach()
            backbone_mask = self._get_feature_vector_attention_mask(
                encoder_backbone_out.shape[1], input_masks
            )

            batch_size, seq_len_enc, enc_dim = encoder_backbone_out.shape
            encoder_backbone_out = encoder_backbone_out.view(batch_size * seq_len_enc, enc_dim)

            distances = (
                encoder_backbone_out.pow(2).sum(1, keepdim=True)
                + self.prototypes.pow(2).sum(1, keepdim=True).transpose(0, 1)
                - 2 * torch.matmul(encoder_backbone_out, self.prototypes.transpose(0, 1))
            )
            # [batch_size, codebook_size]
            codes = torch.argmin(distances, dim=-1)
            codes = codes.reshape(batch_size, seq_len_enc)

            return codes, backbone_mask


class EnCodecTokenizer(nn.Module):
    """KmeansTokenizerWithEncoder"""

    def __init__(self, args):
        super().__init__()

        acoustic_tokenizer_args = self._get_acoustic_tokenizer_args(args)
        self.acoustic_tokenizer_args = acoustic_tokenizer_args

        self.acoustic_tokenizer_args.used_target_bw = getattr(
            acoustic_tokenizer_args, 'used_target_bw', 1
        )
        self.codec = SoundStream(
            n_filters=self.acoustic_tokenizer_args.n_filters,
            D=self.acoustic_tokenizer_args.D,
            target_bandwidths=self.acoustic_tokenizer_args.target_bandwidths,
            ratios=self.acoustic_tokenizer_args.ratios,
            sample_rate=self.acoustic_tokenizer_args.sample_rate,
            bins=self.acoustic_tokenizer_args.bins,
            normalize=self.acoustic_tokenizer_args.normalize,
        )

        self._load_pretrained_acoustic_tokenizer(self.acoustic_tokenizer_args.pretrained_model)

    def _load_pretrained_acoustic_tokenizer(self, acoustic_tokenizer_pretrained_model):
        if acoustic_tokenizer_pretrained_model.startswith("hdfs://"):
            local_dir = '/tmp/'
            tmp_model = 'pretrain_encoder_model.pt'
            dist_hdfs_get(acoustic_tokenizer_pretrained_model, local_dir, tmp_model)
            tmp_model = osp.join(local_dir, tmp_model)
            pretrain_encoder_model_path = tmp_model
        else:
            pretrain_encoder_model_path = acoustic_tokenizer_pretrained_model
        pretrain_encoder_model = torch.load(pretrain_encoder_model_path, map_location='cpu')

        logging.info("Loaded pretrained EnCodec from {}".format(pretrain_encoder_model_path))

        state_dict = {}
        for k, v in pretrain_encoder_model.items():
            k = k.replace("module.", "")
            state_dict[k] = v

        msg = self.codec.load_state_dict(state_dict, strict=True)
        err_msg = []
        if msg.unexpected_keys:
            err_msg.append(
                f'unexpected key in source state_dict: {", ".join(msg.unexpected_keys)}\n'
            )
        if msg.missing_keys:
            err_msg.append(f'missing keys in source state_dict: {", ".join(msg.missing_keys)}\n')
        if err_msg:
            err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
            logging.warning('\n'.join(err_msg))

    def _get_acoustic_tokenizer_args(self, args):
        internal_args = copy.deepcopy(args)
        acoustic_tokenizer_config = ConfigDict()
        for k, v in internal_args.items():
            if k.startswith("acoustic_tokenizer_"):
                new_k = k.replace("acoustic_tokenizer_", "")
                logging.info("Modify arg {} to {}".format(k, new_k))
                acoustic_tokenizer_config[new_k] = v
        return acoustic_tokenizer_config

    def forward(self, inputs, input_masks):
        """
        Forward.
        inputs are waveforms
        """
        with torch.no_grad():
            self.eval()
            inputs /= 32768
            inputs = inputs.unsqueeze(1)  # Adding a channel dim
            codes = self.codec.encode(
                inputs, target_bw=self.acoustic_tokenizer_args.used_target_bw
            )  # [n_q, bsz, seq_len]
            n_q, bsz, seq_len = codes.shape
            device = codes.device
            codes = codes.permute(1, 2, 0)
            for i in range(n_q):
                codes[:, :, i] += i * self.codec.quantizer.bins

            codes = codes.reshape(bsz, seq_len * n_q)
            lengths = input_masks.sum(dim=1)
            subsample_rate = (
                self.codec.quantizer.get_bandwidth_per_quantizer(
                    self.acoustic_tokenizer_args.sample_rate
                )
                * n_q
            )
            subsampled_lengths = (lengths / subsample_rate).long()
            code_masks = torch.zeros_like(codes, device=device).long()
            for i in range(bsz):
                code_masks[i, : subsampled_lengths[i]] = 1
            codes = codes * code_masks
            return codes, code_masks
