''' LSTMLASDecoder '''
import math
import torch
from torch import nn
import torch.nn.functional as F
from core.models.asr.las_attention import MHAttention  # pylint: disable=unused-import
from core.models.layers.transformer import TransformerDecoderLayer
from core.models.layers.embedding import AbsPositionalEncoding, SinusoidalPositionalEmbedding


TRANSFORMER_STATE_SIZE = 6


class LSTMLASDecoder(nn.Module):
    '''
    LSTM LAS decoder
    '''

    def __init__(self, args):
        super().__init__()
        self.update_steps = 0
        self.schedule_sample_begin = args.schedule_sample_begin
        self.schedule_sample_increase_steps = args.schedule_sample_increase_steps
        self.schedule_ratio = args.schedule_ratio

        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.embedding_size)
        self.encoder_proj_fc = nn.Linear(args.backbone_memory_size, args.atten_hidden_size)
        self.attention = eval(args.las_atten_type)(args)

        self.lstm = [
            nn.LSTMCell(
                args.decoder_input_size + args.embedding_size, args.decoder_lstm_hidden_size
            )
        ]
        self.lstm += [
            nn.LSTMCell(args.decoder_lstm_hidden_size, args.decoder_lstm_hidden_size)
            for _ in range(1, args.decoder_lstm_layer_num)
        ]
        self.lstm = nn.Sequential(*self.lstm)
        tgt_vocab_size = args.get('las_tgt_vocab_size', args.tgt_vocab_size)
        # TODO(zhangjun) move model
        self.pred_fc = nn.Sequential(
            *[
                nn.Linear(
                    args.decoder_lstm_hidden_size + args.decoder_input_size + args.embedding_size,
                    args.decoder_pred_fc_proj_size,
                ),
                nn.Tanh(),
                nn.Linear(args.decoder_pred_fc_proj_size, args.decoder_pred_fc_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(args.decoder_dropout),
                nn.Linear(args.decoder_pred_fc_size, args.decoder_pred_fc_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(args.decoder_dropout),
                nn.Linear(args.decoder_pred_fc_size, args.decoder_pred_fc_proj_size, bias=False),
                nn.Linear(args.decoder_pred_fc_proj_size, tgt_vocab_size),
            ]
        )
        self.decoder_lstm_hidden_size = args.decoder_lstm_hidden_size
        self.decoder_lstm_layer_num = args.decoder_lstm_layer_num
        self.incremental_states = None

    def init_states(self, bsz, enc_len=512):
        '''init_states'''
        param = next(self.parameters())
        lstms_state = [
            (
                param.data.new(bsz, self.decoder_lstm_hidden_size).zero_(),
                param.data.new(bsz, self.decoder_lstm_hidden_size).zero_(),
            )
            for _ in range(self.decoder_lstm_layer_num)
        ]
        att_state = self.attention.init_states(bsz, enc_len)
        return [lstms_state, att_state]

    def forward(self, encoder_out, mask, prev_tgt):
        """forward"""
        self.update_steps = self.update_steps + 1
        encoder_proj = self.encoder_proj_fc(encoder_out)
        encoder_mask = mask
        (bsz, enc_len, _) = encoder_out.size()
        lstms_state, att_state = self.init_states(bsz, enc_len)
        prev_emb = self.embed_tokens(prev_tgt)
        tgt_len = prev_tgt.size(1)

        concat_out_list = []

        for token_idx in range(tgt_len):
            att_ctx, att_state = self.attention(
                encoder_out,
                encoder_proj,
                prev_emb[:, token_idx, :],
                lstms_state[-1][0],
                encoder_mask,
                att_state,
            )
            lstm_input = torch.cat([att_ctx, prev_emb[:, token_idx, :]], dim=1)
            new_lstms_state = []
            for layer in range(self.decoder_lstm_layer_num):
                lstm_state = self.lstm[layer](lstm_input, lstms_state[layer])
                lstm_input = lstm_state[0]
                new_lstms_state.append(lstm_state)
            concat_out = torch.cat(
                [prev_emb[:, token_idx, :], att_ctx, new_lstms_state[-1][0]], dim=1
            )
            concat_out_list.append(concat_out.unsqueeze(1))
            lstms_state = new_lstms_state

            # schedule sampling
            if not self.training:
                continue
            assert (
                self.schedule_sample_increase_steps > 0
            ), "schedule_sample_increase_steps must be greater than 0"
            if self.update_steps > self.schedule_sample_begin:
                schedule_ratio = (
                    self.update_steps - self.schedule_sample_begin
                ) / self.schedule_sample_increase_steps
                schedule_ratio = min(schedule_ratio, self.schedule_ratio)
                with torch.no_grad():
                    curr_logits = self.pred_fc(concat_out)
                    hyp = torch.multinomial(F.softmax(curr_logits), num_samples=1).view(bsz)
                    schedule_mask = (
                        (torch.randint(0, 100, (bsz, 1)) > int(100 * schedule_ratio)).cuda().float()
                    )
                hyp_emb = self.embed_tokens(hyp)
                if token_idx < (tgt_len - 1):
                    prev_emb[:, token_idx + 1, :] = prev_emb[
                        :, token_idx + 1, :
                    ] * schedule_mask + hyp_emb * (1 - schedule_mask)

        total_concat_out = torch.cat(concat_out_list, dim=1)
        logits = self.pred_fc(total_concat_out)
        return logits

    def reorder_incremental_state(self, new_order):
        '''
        reorder decoder state for beam selection
        '''
        if self.incremental_states is not None:
            for state_idx, state in enumerate(self.incremental_states):
                if isinstance(state, list):
                    reorder_state = []
                    for layer_state in state:
                        reorder_state.append(
                            (
                                layer_state[0].index_select(0, new_order),
                                layer_state[1].index_select(0, new_order),
                            )
                        )
                else:
                    reorder_state = state.index_select(0, new_order)
                self.incremental_states[state_idx] = reorder_state

    @staticmethod
    def get_normalized_probs(decoder_out, log_probs=True):
        '''get_normalized_probs'''
        if log_probs:
            decoder_out = F.log_softmax(decoder_out.float(), dim=-1)
        else:
            decoder_out = F.softmax(decoder_out.float(), dim=1)
        return decoder_out

    def step(self, tokens, encoder_out, encoder_mask, temperature=1.0, log_probs=True):
        '''
        for las beam search
        '''
        # not so efficient, each time generate encoder_proj
        encoder_proj = self.encoder_proj_fc(encoder_out)

        (bsz, enc_len, _) = encoder_out.size()
        if self.incremental_states is None:
            self.incremental_states = self.init_states(bsz, enc_len)
        lstms_state, att_state = self.incremental_states

        curr_prev_token = tokens[:, -1]
        prev_emb = self.embed_tokens(curr_prev_token)

        att_ctx, att_state = self.attention(
            encoder_out, encoder_proj, prev_emb, lstms_state[-1][0], encoder_mask, att_state
        )
        lstm_input = torch.cat([att_ctx, prev_emb], dim=1)
        new_lstms_state = []
        for layer in range(self.decoder_lstm_layer_num):
            lstm_state = self.lstm[layer](lstm_input, lstms_state[layer])
            lstm_input = lstm_state[0]
            new_lstms_state.append(lstm_state)
        concat_out = torch.cat([prev_emb, att_ctx, new_lstms_state[-1][0]], dim=1)
        logits = self.pred_fc(concat_out)
        self.incremental_states = [new_lstms_state, att_state]

        # temperature
        if temperature != 1.0:
            logits.div_(temperature)
        probs = self.get_normalized_probs(logits, log_probs=log_probs)

        return probs, att_state[:, 1, :]

    def forward_mwer(self, encoder_out, encoder_out_mask, input_dict, reverse=False):
        '''
        for mwer training, get nbest probs
        '''

        encoder_proj = self.encoder_proj_fc(encoder_out)

        (bsz, enc_len, enc_dim) = encoder_out.size()
        if not reverse:
            prev_tgt = input_dict['prev_nbest_sample']  # B, N_sample, tgt_len
        else:
            prev_tgt = input_dict['prev_nbest_sample_rev']
        _, n_sample, tgt_len = prev_tgt.size()
        prev_tgt = prev_tgt.view(-1, tgt_len)
        prev_emb = self.embed_tokens(prev_tgt)
        # B, N_sample, tgt_len, dim
        encoder_out = torch.stack([encoder_out] * n_sample, dim=1).view(-1, enc_len, enc_dim)
        encoder_proj = torch.stack([encoder_proj] * n_sample, dim=1).view(
            -1, encoder_proj.size(1), encoder_proj.size(2)
        )
        encoder_out_mask = torch.stack([encoder_out_mask] * n_sample, dim=1).view(
            -1, encoder_out_mask.size(1)
        )
        lstms_state, att_state = self.init_states(bsz * n_sample, enc_len)

        concat_out_list = []
        for token_idx in range(tgt_len):
            att_ctx, att_state = self.attention(
                encoder_out,
                encoder_proj,
                prev_emb[:, token_idx, :],
                lstms_state[-1][0],
                encoder_out_mask,
                att_state,
            )
            lstm_input = torch.cat([att_ctx, prev_emb[:, token_idx, :]], dim=1)
            new_lstms_state = []
            for layer in range(self.decoder_lstm_layer_num):
                lstm_state = self.lstm[layer](lstm_input, lstms_state[layer])
                lstm_input = lstm_state[0]
                new_lstms_state.append(lstm_state)
            concat_out = torch.cat(
                [prev_emb[:, token_idx, :], att_ctx, new_lstms_state[-1][0]], dim=1
            )
            concat_out_list.append(concat_out.unsqueeze(1))
            lstms_state = new_lstms_state
        total_concat_out = torch.cat(concat_out_list, dim=1).view(bsz, n_sample, tgt_len, -1)
        logits = self.pred_fc(total_concat_out)  # B, N_sample, tgt_len, tgt_dim
        return logits


class LSTMLASDecoderOnnx(nn.Module):
    '''
    LSTM LAS decoder for onnx export
    1. replace lstmcell by lstm to export onnx. Using lstmcell during training for efficiency,
    and use lstm for onnx export. Othewise, error occurs:
        'Exporting the operator _thnn_fused_lstm_cell to ONNX opset version 11 is not supported'
    '''

    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.embedding_size)
        self.encoder_proj_fc = nn.Linear(args.backbone_memory_size, args.atten_hidden_size)

        self.attention = eval(args.las_atten_type)(args)
        self.lstm = [
            nn.LSTM(
                args.decoder_input_size + args.embedding_size,
                args.decoder_lstm_hidden_size,
                batch_first=True,
            )
        ]
        self.lstm += [
            nn.LSTM(args.decoder_lstm_hidden_size, args.decoder_lstm_hidden_size, batch_first=True)
            for _ in range(1, args.decoder_lstm_layer_num)
        ]
        self.lstm = nn.ModuleList(self.lstm)

        tgt_vocab_size = args.get('las_tgt_vocab_size', args.tgt_vocab_size)
        self.pred_fc = nn.Sequential(
            *[
                nn.Linear(
                    args.decoder_lstm_hidden_size + args.backbone_memory_size + args.embedding_size,
                    args.decoder_pred_fc_proj_size,
                ),
                nn.Tanh(),
                nn.Linear(args.decoder_pred_fc_proj_size, args.decoder_pred_fc_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(args.decoder_dropout),
                nn.Linear(args.decoder_pred_fc_size, args.decoder_pred_fc_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(args.decoder_dropout),
                nn.Linear(args.decoder_pred_fc_size, args.decoder_pred_fc_proj_size, bias=False),
                nn.Linear(args.decoder_pred_fc_proj_size, tgt_vocab_size),
            ]
        )

        self.decoder_lstm_layer_num = args.decoder_lstm_layer_num
        self.decoder_lstm_hidden_size = args.decoder_lstm_hidden_size
        self._register_load_state_dict_pre_hook(self.compatible_load_hook)

    def forward(self):
        """forward"""

    @staticmethod
    def compatible_load_hook(
        state_dict, prefix, _local_metadata, _strict, _missing_keys, _unexpected_keys, error_msgs
    ):
        '''
        compatible for load LSTM decoder.
        '''
        for key in list(state_dict.keys()):
            name = prefix + 'lstm'
            if name in key:
                new_key = key + '_l0'
                val = state_dict.pop(key, None)
                # copy lstm1.weight_ih to lstm1.weight_ih_l0
                if state_dict.get(new_key, None) is not None:
                    error_msgs.append(
                        'Both {0} and {1} exist, {1} will be overrided'.format(key, new_key)
                    )
                state_dict[new_key] = val


class SimpleLSTMLASDecoder(LSTMLASDecoder):
    '''
    Simple LSTM LAS decoder
    '''

    def __init__(self, args):
        super().__init__(args)
        self.pred_fc = nn.Sequential(
            *[
                nn.Linear(
                    args.decoder_lstm_hidden_size + args.backbone_memory_size + args.embedding_size,
                    args.decoder_pred_fc_proj_size,
                ),
                nn.Dropout(args.decoder_dropout),
                nn.Linear(args.decoder_pred_fc_proj_size, args.tgt_vocab_size),
            ]
        )


class SimpleLSTMLASDecoderOnnx(LSTMLASDecoderOnnx):
    '''
    LSTM LAS decoder for onnx export
    1. replace lstmcell by lstm to export onnx. Using lstmcell during training for efficiency,
    and use lstm for onnx export. Othewise, error occurs:
        'Exporting the operator _thnn_fused_lstm_cell to ONNX opset version 11 is not supported'
    '''

    def __init__(self, args):
        super().__init__(args)
        self.pred_fc = nn.Sequential(
            *[
                nn.Linear(
                    args.decoder_lstm_hidden_size + args.backbone_memory_size + args.embedding_size,
                    args.decoder_pred_fc_proj_size,
                ),
                nn.Dropout(args.decoder_dropout),
                nn.Linear(args.decoder_pred_fc_proj_size, args.tgt_vocab_size),
            ]
        )


class TransformerLASDecoder(nn.Module):
    '''
    Transformer LAS Decoder
    '''

    def __init__(self, args):
        '''init.'''
        super().__init__()
        self.args = args
        # self.decoder_topology = eval(args.decoder_topology)
        self.decoder_layer_num = args.decoder_layer_num  # len(self.decoder_topology)
        self.states = [[None] * TRANSFORMER_STATE_SIZE] * self.decoder_layer_num
        self.padding_idx = args.tgt_dict.pad()
        self.dropout = args.get('decoder_dropout', 0.0)
        self.decode_residue = args.get('decode_residue', False)
        self.embed_scale = math.sqrt(args.embedding_size)
        self.decoder_pos_embd_type = args.get('decoder_pos_embd_type', 'abs')
        if not args.get('apply_embed_scale', False):
            self.embed_scale = 1.0
        self.transformers = nn.ModuleList()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.embedding_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=args.embedding_size**-0.5)
        if args.embedding_size != args.decoder_embed_dim:
            self.project_in_dim = nn.Linear(args.embedding_size, args.decoder_embed_dim, bias=False)
        else:
            self.project_in_dim = None
        if self.decoder_pos_embd_type == 'abs':
            self.pos_en = AbsPositionalEncoding(args.decoder_embed_dim, args.decoder_dropout)
        else:
            self.pos_en = SinusoidalPositionalEmbedding(
                args.decoder_embed_dim,
                self.padding_idx,
                init_size=args.max_target_positions + self.padding_idx + 1,
            )

        if getattr(self.args, "using_enc_proj_in_dim", False):
            self.enc_proj_in_dim = nn.Linear(
                self.args.backbone_memory_size, self.args.decoder_embed_dim, bias=False
            )

        self.transformers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn=False)
                for _ in range(self.decoder_layer_num)
            ]
        )
        if args.decoder_normalize_before:
            self.final_norm = nn.LayerNorm(args.decoder_embed_dim)
        else:
            self.final_norm = None
        self.pred_fc = nn.Linear(args.decoder_embed_dim, args.tgt_vocab_size, bias=False)

    def reset_state(self):
        '''reset states'''
        self.states = [[None] * TRANSFORMER_STATE_SIZE] * self.decoder_layer_num

    def reorder_incremental_state(self, new_order):
        '''reorder_incremental_state'''
        new_state = []
        for state in self.states:
            st = []
            for i, buffer in enumerate(state):
                if buffer is not None and i in (0, 1):
                    buffer = buffer.index_select(0, new_order)
                st.append(buffer)
            new_state.append(st)
        self.states = new_state

    @staticmethod
    def get_normalized_probs(decoder_out, log_probs=True):
        '''get_normalized_probs'''
        if log_probs:
            decoder_out = F.log_softmax(decoder_out.float(), dim=-1)
        else:
            decoder_out = F.softmax(decoder_out.float(), dim=1)
        return decoder_out

    def prepare_for_onnx_export_(self):
        '''prepare_for_onnx_export_'''
        max_length = 1000  # maximum allowed output symbols length for onnx
        for layer in self.transformers:
            layer.prepare_for_onnx_export_()
            layer.self_attn.prepare_for_onnx_export_()
            layer.encoder_attn.prepare_for_onnx_export_()
        self._future_mask = torch.triu(
            torch.zeros(max_length, max_length).float().cuda().fill_(float("-inf")), 1
        )

    def buffered_future_mask(self, tensor):
        # pylint: disable = access-member-before-definition
        '''prepare decoder self attention mask for causal prediction'''
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                tensor.new(dim, dim).float().fill_(1.0).type_as(tensor), 1
            )
        return self._future_mask[:dim, :dim].bool()

    def forward(self, encoder_out, encoder_mask, prev_tgt, prev_tgt_mask=None):
        '''forward'''
        prev_emb = self.embed_scale * self.embed_tokens(prev_tgt)
        if self.project_in_dim is not None:
            prev_emb = self.project_in_dim(prev_emb)
        if self.decoder_pos_embd_type == 'abs':
            pos_out = self.pos_en(prev_emb)
        else:
            pos_out = self.pos_en(prev_tgt)
        transformer_input = pos_out
        if self.decode_residue:
            transformer_input = transformer_input + prev_emb
        transformer_input = transformer_input.transpose(0, 1)
        if self.args.using_enc_proj_in_dim:
            encoder_out = self.enc_proj_in_dim(encoder_out)
        encoder_out = encoder_out.transpose(0, 1)
        encoder_mask = ~(encoder_mask.bool())
        if prev_tgt_mask is not None:
            prev_tgt_mask = (1 - prev_tgt_mask).int().bool()
        else:
            prev_tgt_mask = torch.zeros_like(prev_tgt).bool()

        self_attn_mask = self.buffered_future_mask(transformer_input)

        transformer_input = F.dropout(transformer_input, p=self.dropout)

        for transformer_layer in self.transformers:
            transformer_input, _ = transformer_layer(
                transformer_input,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=prev_tgt_mask,
            )
        if self.final_norm:
            transformer_input = self.final_norm(transformer_input)
        logits = self.pred_fc(transformer_input).transpose(0, 1)
        return logits

    def forward_step(
        self, tokens, encoder_out, encoder_mask, temperature=1.0, log_probs=True, streaming=True
    ):
        '''
        for las beam search
        '''
        (_bsz, _enc_len, _) = encoder_out.size()
        pos_out = self.pos_en(tokens)
        tokens = tokens[:, -1:]
        pos_out = pos_out[:, -1:]
        prev_emb = self.embed_scale * self.embed_tokens(tokens)
        if self.project_in_dim is not None:
            prev_emb = self.project_in_dim(prev_emb)
        pos_out = pos_out[:, -1:, :]
        transformer_input = prev_emb + pos_out
        transformer_input = transformer_input.transpose(0, 1)
        if self.args.using_enc_proj_in_dim:
            encoder_out = self.enc_proj_in_dim(encoder_out)
        encoder_out = encoder_out.transpose(0, 1)
        encoder_mask = encoder_mask.bool()
        encoder_mask = ~encoder_mask

        if streaming:
            for idx in range(TRANSFORMER_STATE_SIZE):
                if self.states[idx][-1] is not None:
                    self.states[idx][-1] = encoder_mask.type(torch.bool)

        self_attn_padding_mask = None
        if tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = tokens.eq(self.padding_idx)

        for idx, transformer_layer in enumerate(self.transformers):
            self_attn_mask = None
            transformer_input, att_state, cache = transformer_layer.forward_step(
                transformer_input,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_mask,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                state=self.states[idx],
            )
            self.states[idx] = cache
        if self.final_norm:
            transformer_input = self.final_norm(transformer_input)
        logits = self.pred_fc(transformer_input).transpose(0, 1).squeeze()

        # temperature
        if temperature != 1.0:
            logits.div_(temperature)
        probs = self.get_normalized_probs(logits, log_probs=log_probs)

        return probs, att_state.squeeze()


class LSTMDelibDecoder(LSTMLASDecoder):
    """LAS Decoder With Deliberation"""

    def __init__(self, args):
        super().__init__(args)
        self.las_hyp_encoder_proj_fc = nn.Linear(args.backbone_memory_size, args.atten_hidden_size)
        self.las_hyp_attention = eval(args.las_atten_type)(args)

    def init_states(self, bsz, enc_len, hyp_len):
        '''init_states'''
        lstms_state, acoustic_att_state = super().init_states(bsz, enc_len)
        hyp_att_state = self.las_hyp_attention.init_states(bsz, hyp_len)
        return [lstms_state, acoustic_att_state, hyp_att_state]

    def forward(self, encoder_outs, encoder_out_masks, prev_tgt):
        # pylint: disable=too-many-locals
        acoustic_encoder_out, hyp_encoder_out = encoder_outs
        acoustic_mask, hyp_mask = encoder_out_masks
        self.update_steps = self.update_steps + 1
        hyp_encoder_proj = self.las_hyp_encoder_proj_fc(hyp_encoder_out)
        acoustic_encoder_proj = self.encoder_proj_fc(acoustic_encoder_out)
        (bsz, hyp_len, _) = hyp_encoder_out.size()
        acoustic_len = acoustic_encoder_out.size(1)
        lstms_state, acoustic_att_state, hyp_att_state = self.init_states(
            bsz, acoustic_len, hyp_len
        )
        prev_emb = self.embed_tokens(prev_tgt)
        tgt_len = prev_tgt.size(1)

        concat_out_list = []

        for token_idx in range(tgt_len):
            hyp_att_ctx, hyp_att_state = self.las_hyp_attention(
                hyp_encoder_out,
                hyp_encoder_proj,
                prev_emb[:, token_idx, :],
                lstms_state[-1][0],
                hyp_mask,
                hyp_att_state,
            )
            acoustic_att_ctx, acoustic_att_state = self.attention(
                acoustic_encoder_out,
                acoustic_encoder_proj,
                prev_emb[:, token_idx, :],
                lstms_state[-1][0],
                acoustic_mask,
                acoustic_att_state,
            )
            lstm_input = torch.cat(
                [hyp_att_ctx, acoustic_att_ctx, prev_emb[:, token_idx, :]], dim=1
            )
            new_lstms_state = []
            for layer in range(self.decoder_lstm_layer_num):
                lstm_state = self.lstm[layer](lstm_input, lstms_state[layer])
                lstm_input = lstm_state[0]
                new_lstms_state.append(lstm_state)
            concat_out = torch.cat(
                [prev_emb[:, token_idx, :], hyp_att_ctx, acoustic_att_ctx, new_lstms_state[-1][0]],
                dim=1,
            )
            concat_out_list.append(concat_out.unsqueeze(1))
            lstms_state = new_lstms_state

            # schedule sampling
            if not self.training:
                continue
            assert (
                self.schedule_sample_increase_steps > 0
            ), "schedule_sample_increase_steps must be greater than 0"
            if self.update_steps > self.schedule_sample_begin:
                schedule_ratio = (
                    self.update_steps - self.schedule_sample_begin
                ) / self.schedule_sample_increase_steps
                schedule_ratio = min(schedule_ratio, self.schedule_ratio)
                with torch.no_grad():
                    curr_logits = self.pred_fc(concat_out)
                    hyp = torch.multinomial(F.softmax(curr_logits), num_samples=1).view(bsz)
                    schedule_mask = (
                        (torch.randint(0, 100, (bsz, 1)) > int(100 * schedule_ratio)).cuda().float()
                    )
                hyp_emb = self.embed_tokens(hyp)
                if token_idx < (tgt_len - 1):
                    prev_emb[:, token_idx + 1, :] = prev_emb[
                        :, token_idx + 1, :
                    ] * schedule_mask + hyp_emb * (1 - schedule_mask)

        total_concat_out = torch.cat(concat_out_list, dim=1)
        logits = self.pred_fc(total_concat_out)
        return logits

    def forward_step(
        self, tokens, encoder_outs, encoder_out_masks, temperature=1.0, log_probs=True, **_kwargs
    ):
        '''
        for multi-stream las beam search
        '''
        # not so efficient, each time generate encoder_proj
        acoustic_encoder_out, hyp_encoder_out = encoder_outs
        acoustic_mask, hyp_mask = encoder_out_masks
        hyp_encoder_proj = self.las_hyp_encoder_proj_fc(hyp_encoder_out)
        acoustic_encoder_proj = self.encoder_proj_fc(acoustic_encoder_out)

        (bsz, hyp_len, _) = hyp_encoder_out.size()
        acoustic_len = acoustic_encoder_out.size(1)
        if self.incremental_states is None:
            self.incremental_states = self.init_states(bsz, acoustic_len, hyp_len)
        lstms_state, acoustic_att_state, hyp_att_state = self.incremental_states

        curr_prev_token = tokens[:, -1]
        prev_emb = self.embed_tokens(curr_prev_token)

        hyp_att_ctx, hyp_att_state = self.las_hyp_attention(
            hyp_encoder_out, hyp_encoder_proj, prev_emb, lstms_state[-1][0], hyp_mask, hyp_att_state
        )
        acoustic_att_ctx, acoustic_att_state = self.attention(
            acoustic_encoder_out,
            acoustic_encoder_proj,
            prev_emb,
            lstms_state[-1][0],
            acoustic_mask,
            acoustic_att_state,
        )
        lstm_input = torch.cat([hyp_att_ctx, acoustic_att_ctx, prev_emb], dim=1)
        new_lstms_state = []
        for layer in range(self.decoder_lstm_layer_num):
            lstm_state = self.lstm[layer](lstm_input, lstms_state[layer])
            lstm_input = lstm_state[0]
            new_lstms_state.append(lstm_state)
        concat_out = torch.cat(
            [prev_emb, hyp_att_ctx, acoustic_att_ctx, new_lstms_state[-1][0]], dim=1
        )
        logits = self.pred_fc(concat_out)
        self.incremental_states = [new_lstms_state, acoustic_att_state, hyp_att_state]

        # temperature
        if temperature != 1.0:
            logits.div_(temperature)
        probs = self.get_normalized_probs(logits, log_probs=log_probs)

        return probs, hyp_att_state[:, 1, :]
