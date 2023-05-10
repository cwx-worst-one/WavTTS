"""
ce_encoder.py.
"""
import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.fsmn_layer import FSMNLayer
from core.models.layers.lstmp_layer import LSTMP
from core.models.layers.unfold import PantherUnFold
from core.models.asr.acoustic_backbone import DFSMNBackboneLN
from core.models.sid.frame_level_net import ResnetBackbone
from core.models.sid.pooling_net import StatPooling
from core.models.sid.segment_level_net import SegmentMLP

# pylint:disable=unused-import
from core.models.vad.acoustic_backbone import ConformerBackbone, MaskedConformerBackbone


class DFSMNEncoder(nn.Module):
    """DFSMN encoder"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_trans_fc = nn.Sequential(
            *[
                nn.Linear(args.fbank_dim * args.input_concat_size, args.dfsmn_hidden_size),
                nn.ReLU(),
                nn.Linear(args.dfsmn_hidden_size, args.dfsmn_memory_size, bias=False),
            ]
        )
        self.fsmn = FSMNLayer(
            args.dfsmn_memory_size,
            args.fsmn_left_kernel_size,
            args.fsmn_right_kernel_size,
            dilation=args.fsmn_dilation,
        )
        self.dfsmn = DFSMNBackboneLN(args)
        self.pred_fc = torch.nn.Sequential(
            *[
                torch.nn.Linear(args.dfsmn_memory_size, args.dfsmn_hidden_size),
                torch.nn.ReLU(),
                nn.Dropout(args.dfsmn_dropout),
                torch.nn.Linear(args.dfsmn_hidden_size, args.tgt_vocab_size),
            ]
        )

        self.unfold = PantherUnFold(
            (args.input_concat_size, 1), stride=(args.downsampling_size, 1), padding=(0, 0)
        )
        self.input_concat_size = args.input_concat_size
        self.fbank_dim = args.fbank_dim
        self.selected_dfsmn_layer = args.get("selected_dfsmn_layer", -1)
        self.do_cmvn = args.get("model_do_cmvn", False)
        if self.do_cmvn:
            self.mean = nn.Parameter(torch.zeros(args.fbank_dim))
            self.inv_std = nn.Parameter(torch.ones(args.fbank_dim))

    def forward(self, fbank, fbank_mask=None, ce_label=None):
        """forward"""
        if self.do_cmvn:
            fbank = (fbank - self.mean) * self.inv_std
        bsz = fbank.shape[0]
        unfold_fbank = (
            (self.unfold(fbank.unsqueeze(1)))
            .contiguous()
            .view(bsz, self.input_concat_size, -1, self.fbank_dim)
        )
        unfold_fbank = (
            unfold_fbank.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.input_concat_size * self.fbank_dim)
        )
        hard_mask = expand_hard_mask = fbank_mask
        if hard_mask is not None:
            dfsmn_mask = self.unfold(fbank_mask.unsqueeze(1).unsqueeze(3))
            hard_mask = dfsmn_mask[:, self.input_concat_size - 1, :].contiguous()
            expand_hard_mask = hard_mask.unsqueeze(2)
        input_trans = self.input_trans_fc(unfold_fbank)
        fsmn_out = self.fsmn(input_trans, expand_hard_mask)
        layer_out = self.dfsmn(
            fsmn_out,
            expand_hard_mask.squeeze(2),
            "BTN",
            selected_layer_idx=self.selected_dfsmn_layer,
        )
        if self.selected_dfsmn_layer != -1:
            dfsmn_out, selected_dfsmn_out = layer_out
        else:
            dfsmn_out = layer_out
            selected_dfsmn_out = None
        encoder_out = self.pred_fc(dfsmn_out)

        tgt = hard_tgt = ce_label
        if tgt is not None:
            tgt = tgt.type_as(fbank).unsqueeze(1).unsqueeze(3)
            unfold_tgt = self.unfold(tgt)  # (B,downsample_size,T)
            hard_tgt = unfold_tgt[:, self.args.downsampling_size // 2, :].contiguous().long()

        return encoder_out, selected_dfsmn_out, hard_mask, hard_tgt


class ConformerEncoder(nn.Module):
    """Conformer encoder"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.conformer = eval(args.acoustic_backbone_type)(args)

        self.input_trans_fc = nn.Sequential(
            *[
                nn.Linear(args.fbank_dim * args.input_concat_size, args.backbone_memory_size),
                nn.ReLU(),
                nn.Linear(args.backbone_memory_size, args.backbone_memory_size, bias=False),
            ]
        )

        self.pred_fc = torch.nn.Sequential(
            *[
                torch.nn.Linear(args.backbone_memory_size, args.backbone_hidden_size),
                torch.nn.ReLU(),
                nn.Dropout(args.conformer_linearPredict_dropout),
                torch.nn.Linear(args.backbone_hidden_size, args.tgt_vocab_size),
            ]
        )

        self.unfold = PantherUnFold(
            (args.input_concat_size, 1), stride=(args.downsampling_size, 1), padding=(0, 0)
        )
        self.input_concat_size = args.input_concat_size
        self.fbank_dim = args.fbank_dim

        self.do_cmvn = args.get("model_do_cmvn", False)
        if self.do_cmvn:
            self.mean = nn.Parameter(torch.zeros(args.fbank_dim))
            self.inv_std = nn.Parameter(torch.ones(args.fbank_dim))

    def forward(self, fbank, fbank_mask=None, ce_label=None):
        """forward"""
        if self.do_cmvn:
            fbank = (fbank - self.mean) * self.inv_std
        bsz = fbank.shape[0]
        unfold_fbank = (
            (self.unfold(fbank.unsqueeze(1)))
            .contiguous()
            .view(bsz, self.input_concat_size, -1, self.fbank_dim)
        )
        unfold_fbank = (
            unfold_fbank.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.input_concat_size * self.fbank_dim)
        )
        hard_mask = expand_hard_mask = fbank_mask
        if hard_mask is not None:
            dfsmn_mask = self.unfold(fbank_mask.unsqueeze(1).unsqueeze(3))
            hard_mask = dfsmn_mask[:, self.input_concat_size - 1, :].contiguous()
            expand_hard_mask = hard_mask.unsqueeze(2)
        input_trans = self.input_trans_fc(unfold_fbank)
        conformer_out = self.conformer(input_trans, expand_hard_mask, frontend_shape="BTN")
        encoder_out = self.pred_fc(conformer_out)

        tgt = hard_tgt = ce_label
        if tgt is not None:
            tgt = tgt.type_as(fbank).unsqueeze(1).unsqueeze(3)
            unfold_tgt = self.unfold(tgt)  # (B,downsample_size,T)
            hard_tgt = unfold_tgt[:, self.args.downsampling_size // 2, :].contiguous().long()

        return encoder_out, None, hard_mask, hard_tgt


class TsVadEncoder(nn.Module):
    """Target speaker VAD encoder"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.front_end_resnet = ResnetBackbone(args)
        self.front_end_gsp = StatPooling(args, self.front_end_resnet.output_dim)
        self.front_end_linear = SegmentMLP(args, self.front_end_gsp.output_dim)

        # Transformer encoder
        if args.tsvad_encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.front_end_linear.output_dim * 2,
                nhead=args.encoder_attn_heads,
                dim_feedforward=args.encoder_ffn_embed_dim,
                dropout=args.encoder_dropout,
                activation=args.encoder_self_attn_activation_fn,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.encoder_num_layers)
            encoder_output_dim = self.front_end_linear.output_dim * 2
        elif args.tsvad_encoder_type == "blstm":
            self.encoder = nn.ModuleList()
            input_size = self.front_end_linear.output_dim * 2
            for _ in range(args.encoder_num_layers):
                self.encoder.append(
                    LSTMP(
                        input_size,
                        args.encoder_hidden_size,
                        output_size=args.encoder_output_size,
                        batch_first=True,
                        bidirectional=True,
                        dropout=args.encoder_dropout,
                        residual=args.encoder_residual,
                    )
                )
                input_size = args.encoder_output_size
            encoder_output_dim = args.encoder_output_size
        else:
            raise NotImplementedError(
                "tsvad_encoder_type = {} not implemented".format(args.tsvad_encoder_type)
            )

        # BiLSTM decoder
        self.decoder = nn.LSTM(
            encoder_output_dim * args.target_speaker_num,
            args.lstm_hidden_size,
            args.lstm_num_layers,
            dropout=args.lstm_dropout,
            bidirectional=args.lstm_bidirectional,
            batch_first=True,
        )
        decoder_output_dim = args.lstm_hidden_size * (2 if args.lstm_bidirectional else 1)

        proj_output_dim = decoder_output_dim
        if args.get("proj_size", None) is not None:
            proj_size = eval(args.proj_size)
            self.proj_layers = nn.ModuleList()
            for num_node in proj_size:
                self.proj_layers.append(nn.Linear(proj_output_dim, num_node))
                self.proj_layers.append(
                    nn.BatchNorm1d(
                        num_node,
                        momentum=args.batchnorm_momentum,
                        eps=args.batchnorm_eps,
                    )
                )
                self.proj_layers.append(nn.ReLU(inplace=True))
                proj_output_dim = num_node

        self.ts_detections = nn.Linear(proj_output_dim, args.target_speaker_num)
        self.freeze_frontend = args.get("freeze_frontend", False)

    # pylint: disable=unused-argument
    def forward(self, fbank, ts_embeddings, fbank_mask=None, ts_ce_label=None):
        """forward"""
        if self.freeze_frontend:
            with torch.no_grad():
                frame_level_feat, _ = self.front_end_resnet(fbank)
                gsp, _ = self.front_end_gsp(frame_level_feat)
                input_embeddings, _ = self.front_end_linear(gsp)
        else:
            frame_level_feat, _ = self.front_end_resnet(fbank)
            gsp, _ = self.front_end_gsp(frame_level_feat)
            input_embeddings, _ = self.front_end_linear(gsp)

        if self.args.get("normalize_embedding", True):
            # Normalize the reference and extracted embeddings
            input_embeddings = F.normalize(input_embeddings.transpose(1, 2), dim=-1)
            ts_embeddings = F.normalize(ts_embeddings, dim=-1)
        else:
            input_embeddings = input_embeddings.transpose(1, 2)

        bsz, tsz, dim = input_embeddings.shape
        nspks = ts_embeddings.shape[1]
        # input_embeddings: [B, T, E] -> [B, C, T, E]
        repeated_input_embeddings = input_embeddings.unsqueeze(dim=1).expand(-1, nspks, -1, -1)
        # ts_embedding: [B, C, E] -> repeated_ts_embeddings: [B, C, T, E]
        repeated_ts_embeddings = ts_embeddings.unsqueeze(dim=2).expand(-1, -1, tsz, -1)
        concat_embeddings = torch.cat([repeated_input_embeddings, repeated_ts_embeddings], dim=-1)

        # [B, C, T, E] -> [B*C, T, E] -> [T, B*C, E]
        concat_embeddings = concat_embeddings.view(bsz * nspks, tsz, dim * 2)
        if self.args.tsvad_encoder_type == "transformer":
            # The transformer requires the input to be [T, B, E]
            encoded_embeddings = self.encoder(concat_embeddings.transpose(0, 1))
            encoded_embeddings = encoded_embeddings.transpose(0, 1)
        elif self.args.tsvad_encoder_type == "blstm":
            encoded_embeddings = concat_embeddings
            for layer in self.encoder:
                encoded_embeddings = layer(encoded_embeddings)
        else:
            raise NotImplementedError(
                "tsvad_encoder_type = {} not implemented".format(self.args.tsvad_encoder_type)
            )

        # [B*C, T, E] -> [B, C, T, E] -> [B, T, C, E] -> [B, T, C*E]
        encoded_embeddings = (
            encoded_embeddings.view(bsz, nspks, tsz, -1)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tsz, -1)
        )

        # decoder
        decoded_embeddings, _ = self.decoder(encoded_embeddings)

        # projection
        proj_embeddings = decoded_embeddings
        if self.args.get("proj_size", None) is not None:
            for layer in self.proj_layers:
                if isinstance(layer, nn.BatchNorm1d):
                    proj_embeddings = layer(proj_embeddings.transpose(1, 2)).transpose(1, 2)
                else:
                    proj_embeddings = layer(proj_embeddings)

        # output
        target_logits = self.ts_detections(proj_embeddings)

        # The output is the logits (rather than the probabilities)
        return target_logits


class MultiTsVadEncoder(nn.Module):
    """Target speaker VAD encoder for multi-channel"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.front_end_resnet = ResnetBackbone(args)
        self.front_end_gsp = StatPooling(args, self.front_end_resnet.output_dim)
        self.front_end_linear = SegmentMLP(args, self.front_end_gsp.output_dim)

        # Cross-channel self-attention
        ccsa_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.front_end_linear.output_dim * 2,
            nhead=args.cross_channel_attn_heads,
            dim_feedforward=args.cross_channel_ffn_embed_dim,
            dropout=args.cross_channel_dropout,
            activation=args.cross_channel_self_attn_activation_fn,
        )
        self.ccsa_encoder = nn.TransformerEncoder(
            ccsa_encoder_layer, num_layers=args.cross_channel_encoder_num_layers
        )

        # Transformer encoder
        if args.tsvad_encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.front_end_linear.output_dim * 2,
                nhead=args.encoder_attn_heads,
                dim_feedforward=args.encoder_ffn_embed_dim,
                dropout=args.encoder_dropout,
                activation=args.encoder_self_attn_activation_fn,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.encoder_num_layers)
            encoder_output_dim = self.front_end_linear.output_dim * 2
        elif args.tsvad_encoder_type == "blstm":
            self.encoder = nn.ModuleList()
            input_size = self.front_end_linear.output_dim * 2
            for _ in range(args.encoder_num_layers):
                self.encoder.append(
                    LSTMP(
                        input_size,
                        args.encoder_hidden_size,
                        output_size=args.encoder_output_size,
                        batch_first=True,
                        bidirectional=True,
                        dropout=args.encoder_dropout,
                        residual=args.encoder_residual,
                    )
                )
                input_size = args.encoder_output_size
            encoder_output_dim = args.encoder_output_size
        else:
            raise NotImplementedError(
                f"tsvad_encoder_type = {args.tsvad_encoder_type} not implemented"
            )

        # BiLSTM decoder
        self.decoder = nn.LSTM(
            encoder_output_dim * args.target_speaker_num,
            args.lstm_hidden_size,
            args.lstm_num_layers,
            dropout=args.lstm_dropout,
            bidirectional=args.lstm_bidirectional,
            batch_first=True,
        )
        decoder_output_dim = args.lstm_hidden_size * (2 if args.lstm_bidirectional else 1)

        proj_output_dim = decoder_output_dim
        if args.get("proj_size", None) is not None:
            proj_size = eval(args.proj_size)
            self.proj_layers = nn.ModuleList()
            for num_node in proj_size:
                self.proj_layers.append(nn.Linear(proj_output_dim, num_node))
                self.proj_layers.append(
                    nn.BatchNorm1d(
                        num_node,
                        momentum=args.batchnorm_momentum,
                        eps=args.batchnorm_eps,
                    )
                )
                self.proj_layers.append(nn.ReLU(inplace=True))
                proj_output_dim = num_node

        self.ts_detections = nn.Linear(proj_output_dim, args.target_speaker_num)
        self.freeze_frontend = args.get("freeze_frontend", False)

    # pylint: disable=unused-argument,invalid-name
    def forward(self, fbank, ts_embeddings, fbank_mask=None, ts_ce_label=None):
        """forward
        fbank: [B, C, F, T]
        """
        bsz, csz, fdim, tsz = fbank.shape
        fbank = fbank.reshape([bsz * csz, fdim, tsz])
        if self.freeze_frontend:
            with torch.no_grad():
                frame_level_feat, _ = self.front_end_resnet(fbank)
                gsp, _ = self.front_end_gsp(frame_level_feat)
                input_embeddings, _ = self.front_end_linear(gsp)
        else:
            frame_level_feat, _ = self.front_end_resnet(fbank)
            gsp, _ = self.front_end_gsp(frame_level_feat)
            input_embeddings, _ = self.front_end_linear(gsp)

        if self.args.get("normalize_embedding", True):
            # Normalize the reference and extracted embeddings
            input_embeddings = F.normalize(input_embeddings.transpose(1, 2), dim=-1)
            ts_embeddings = F.normalize(ts_embeddings, dim=-1)
        else:
            input_embeddings = input_embeddings.transpose(1, 2)

        _, tsz, dim = input_embeddings.shape
        input_embeddings = input_embeddings.reshape([bsz, csz, tsz, dim])
        nspks = ts_embeddings.shape[1]
        repeated_input_embeddings = input_embeddings.unsqueeze(dim=1).expand(-1, nspks, -1, -1, -1)
        repeated_ts_embeddings = ts_embeddings.unsqueeze(dim=-2).expand(-1, -1, -1, tsz, -1)
        concat_embeddings = torch.cat([repeated_input_embeddings, repeated_ts_embeddings], dim=-1)

        # [B, N, C, T, E] -> [C, B, N, T, E] -> [C, B*N*T, E]
        concat_embeddings = concat_embeddings.permute(2, 0, 1, 3, 4).reshape(
            csz, bsz * nspks * tsz, dim * 2
        )
        # cross-channel transformer -> [B*N*T, C, E] -> average [B*N*T, E] -> [B*N, T, E]
        encoded_cross_channel_embeddings = self.ccsa_encoder(concat_embeddings).transpose(0, 1)
        encoded_cross_channel_embeddings = torch.mean(encoded_cross_channel_embeddings, axis=1)
        encoded_cross_channel_embeddings = encoded_cross_channel_embeddings.reshape(
            bsz * nspks, tsz, dim * 2
        )

        if self.args.tsvad_encoder_type == "transformer":
            encoded_embeddings = self.encoder(encoded_cross_channel_embeddings.transpose(0, 1))
            encoded_embeddings = encoded_embeddings.transpose(0, 1)
        elif self.args.tsvad_encoder_type == "blstm":
            encoded_embeddings = encoded_cross_channel_embeddings
            for layer in self.encoder:
                encoded_embeddings = layer(encoded_embeddings)
        else:
            raise NotImplementedError(
                f"tsvad_encoder_type = {self.args.tsvad_encoder_type} not implemented"
            )

        # [B*N, T, E] -> [B, N, T, E] -> [B, T, N, E] -> [B, T, N*E]
        encoded_embeddings = (
            encoded_embeddings.view(bsz, nspks, tsz, -1)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tsz, -1)
        )

        # decoder
        decoded_embeddings, _ = self.decoder(encoded_embeddings)

        # projection
        proj_embeddings = decoded_embeddings
        if self.args.get("proj_size", None) is not None:
            for layer in self.proj_layers:
                if isinstance(layer, nn.BatchNorm1d):
                    proj_embeddings = layer(proj_embeddings.transpose(1, 2)).transpose(1, 2)
                else:
                    proj_embeddings = layer(proj_embeddings)

        # output
        target_logits = self.ts_detections(proj_embeddings)

        # The output is the logits (rather than the probabilities)
        return target_logits
