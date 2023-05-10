''' rnnt predictor '''
# pylint:disable=too-many-lines
import torch
from torch import nn

from core.models.layers.embedding import (
    SimpleSinusoidalPositionalEmbedding,
    IncrementalReducedEmbedding,
)
from core.models.layers.lstmp_layer import (
    IncrementalLSTM,
    IncrementalLimitedContextLSTM,
)
from core.extensions import fused_prev_char
from core.models.layers.transformer_xl import RelPartialLearnableDecoderLayer
from core.models.layers.active_function import get_activation_fn


def lstm_select_non_blank_states(
    incremental_states, incremental_states_next, mask, lstm_type='Incremental'
):
    '''lstm_select_non_blank_states'''
    if lstm_type == 'Incremental':
        layer_idx = 0
        for predicter_state, predicter_state_next in zip(
            incremental_states, incremental_states_next
        ):
            predicter_h = torch.where(mask, predicter_state[0], predicter_state_next[0])
            predicter_c = torch.where(mask, predicter_state[1], predicter_state_next[1])
            incremental_states[layer_idx] = [predicter_h, predicter_c]
            layer_idx += 1
    elif lstm_type == 'Pytorch':
        predicter_h = torch.where(mask, incremental_states[0], incremental_states_next[0])
        predicter_c = torch.where(mask, incremental_states[1], incremental_states_next[1])
        incremental_states = (predicter_h, predicter_c)
    elif lstm_type == 'LimitedContext':
        incremental_states = torch.where(mask, incremental_states, incremental_states_next)

    return incremental_states


def lstm_first_step_expand_state(states, beam_size, lstm_type='Incremental'):
    '''first_step_expand_state'''
    if lstm_type == 'Incremental':
        new_states = []
        for state in states:
            h_state, c_state = state
            ndim = h_state.size(-1)
            expand_h_state = h_state.unsqueeze(1).repeat(1, beam_size, 1).contiguous()
            expand_c_state = c_state.unsqueeze(1).repeat(1, beam_size, 1).contiguous()
            new_states.append((expand_h_state.view(-1, ndim), expand_c_state.view(-1, ndim)))
    elif lstm_type == 'Pytorch':
        h_state, c_state = states
        nlayer = h_state.size(0)
        ndim = h_state.size(-1)
        expand_h_state = h_state.unsqueeze(2).repeat(1, 1, beam_size, 1).contiguous()
        expand_c_state = c_state.unsqueeze(2).repeat(1, 1, beam_size, 1).contiguous()
        new_states = (expand_h_state.view(nlayer, -1, ndim), expand_c_state.view(nlayer, -1, ndim))
    elif lstm_type == 'LimitedContext':
        ndim = states.size(-1)
        expand_state = states.unsqueeze(1).repeat(1, beam_size, 1).contiguous()
        new_states = expand_state.view(-1, ndim)
    return new_states


def lstm_first_step_concate_state(init_states, expand_states, beam_size, lstm_type='Incremental'):
    '''first_step_concate_state'''
    if lstm_type == 'Incremental':
        concat_states = []
        for init_state, expand_state in zip(init_states, expand_states):
            init_h_state, init_c_state = init_state
            bsz, ndim = init_h_state.size()
            # init_h_state (B, N)
            expand_h_state, expand_c_state = expand_state
            # expand_h_state (B, beam-1, N)
            concat_h_state = torch.cat(
                [init_h_state.unsqueeze(1), expand_h_state.view(bsz, -1, ndim)], dim=1
            )
            concat_c_state = torch.cat(
                [init_c_state.unsqueeze(1), expand_c_state.view(bsz, -1, ndim)], dim=1
            )
            concat_states.append(
                (
                    concat_h_state.contiguous().view(bsz * beam_size, ndim),
                    concat_c_state.contiguous().view(bsz * beam_size, ndim),
                )
            )
    elif lstm_type == 'Pytorch':
        init_h_state, init_c_state = init_states
        nlayer, bsz, ndim = init_h_state.size()
        # init_h_state (B, N)
        expand_h_state, expand_c_state = expand_states
        # expand_h_state (B, beam-1, N)
        concat_h_state = torch.cat(
            [init_h_state.unsqueeze(2), expand_h_state.view(nlayer, bsz, -1, ndim)], dim=2
        )
        concat_c_state = torch.cat(
            [init_c_state.unsqueeze(2), expand_c_state.view(nlayer, bsz, -1, ndim)], dim=2
        )
        concat_states = (
            concat_h_state.contiguous().view(nlayer, bsz * beam_size, ndim),
            concat_c_state.contiguous().view(nlayer, bsz * beam_size, ndim),
        )
    elif lstm_type == 'LimitedContext':
        bsz, ndim = init_states.size()
        concat_states = torch.cat(
            [init_states.unsqueeze(1), expand_states.view(bsz, -1, ndim)], dim=1
        )
    return concat_states


def lstm_reorder_beam_states(states, beam_idx, lstm_type='Incremental'):
    '''state: (bsz*(#beam), ndim) beam_idx: (bsz, beam) -> out: (bsz*beam, ndim)'''
    bsz = beam_idx.size(0)
    if lstm_type == 'Incremental':
        reorder_states = []
        for state in states:
            h_state, c_state = state
            ndim = h_state.size(-1)
            reorder_h_state = torch.gather(
                h_state.view(bsz, -1, ndim), 1, beam_idx.unsqueeze(2).repeat(1, 1, ndim)
            )
            reorder_c_state = torch.gather(
                c_state.view(bsz, -1, ndim), 1, beam_idx.unsqueeze(2).repeat(1, 1, ndim)
            )
            reorder_states.append(
                (
                    reorder_h_state.contiguous().view(-1, ndim),
                    reorder_c_state.contiguous().view(-1, ndim),
                )
            )
    elif lstm_type == 'Pytorch':
        h_state, c_state = states
        nlayer = h_state.size(0)
        ndim = h_state.size(-1)
        reorder_h_state = torch.gather(
            h_state.view(nlayer, bsz, -1, ndim),
            2,
            beam_idx.unsqueeze(0).unsqueeze(3).repeat(nlayer, 1, 1, ndim),
        )
        reorder_c_state = torch.gather(
            c_state.view(nlayer, bsz, -1, ndim),
            2,
            beam_idx.unsqueeze(0).unsqueeze(3).repeat(nlayer, 1, 1, ndim),
        )
        reorder_states = (
            reorder_h_state.contiguous().view(nlayer, -1, ndim),
            reorder_c_state.contiguous().view(nlayer, -1, ndim),
        )
    elif lstm_type == 'LimitedContext':
        ndim = states.size(-1)
        reorder_states = torch.gather(
            states.view(bsz, -1, ndim),
            1,
            beam_idx.unsqueeze(2).repeat(1, 1, ndim),
        )
        reorder_states = reorder_states.contiguous().view(-1, ndim)
    return reorder_states


def lstm_concate_t_ut_states(t_states, u_t_states, beam_size, lstm_type='Incremental'):
    '''(B*beam, N) * 2 -> (B*(2*beam), N)'''
    if lstm_type == 'Incremental':
        out_states = []
        for t_state, u_t_state in zip(t_states, u_t_states):
            h_t_state, c_t_state = t_state
            ndim = h_t_state.size(-1)
            h_u_t_state, c_u_t_state = u_t_state
            # (B, 2*beam, N)
            concat_h_state = torch.cat(
                [h_t_state.view(-1, beam_size, ndim), h_u_t_state.view(-1, beam_size, ndim)], dim=1
            )
            # (B, 2*beam, N)
            concat_c_state = torch.cat(
                [c_t_state.view(-1, beam_size, ndim), c_u_t_state.view(-1, beam_size, ndim)], dim=1
            )
            out_states.append((concat_h_state.view(-1, ndim), concat_c_state.view(-1, ndim)))
    elif lstm_type == 'Pytorch':
        h_t_state, c_t_state = t_states
        nlayer = h_t_state.size(0)
        ndim = h_t_state.size(-1)
        h_u_t_state, c_u_t_state = u_t_states
        # (nlayer, B, 2*beam, N)
        concat_h_state = torch.cat(
            [
                h_t_state.view(nlayer, -1, beam_size, ndim),
                h_u_t_state.view(nlayer, -1, beam_size, ndim),
            ],
            dim=2,
        )
        # (nlayer, B, 2*beam, N)
        concat_c_state = torch.cat(
            [
                c_t_state.view(nlayer, -1, beam_size, ndim),
                c_u_t_state.view(nlayer, -1, beam_size, ndim),
            ],
            dim=2,
        )
        out_states = (concat_h_state.view(nlayer, -1, ndim), concat_c_state.view(nlayer, -1, ndim))
    elif lstm_type == 'LimitedContext':
        ndim = t_states.size(-1)
        concat_state = torch.cat(
            [
                t_states.view(-1, beam_size, ndim),
                u_t_states.view(-1, beam_size, ndim),
            ],
            dim=1,
        )
        out_states = concat_state.view(-1, ndim)
    return out_states


def lstm_concate_t_ut_uut_states(
    t_states, u_t_states, u_u_t_states, beam_size, lstm_type='Incremental'
):
    '''(B*beam, N) * 3 -> (B*(3*beam), N)'''
    if lstm_type == 'Incremental':
        out_states = []
        for t_state, u_t_state, u_u_t_state in zip(t_states, u_t_states, u_u_t_states):
            h_t_state, c_t_state = t_state
            ndim = h_t_state.size(-1)
            h_u_t_state, c_u_t_state = u_t_state
            h_u_u_t_state, c_u_u_t_state = u_u_t_state
            # (B, 3*beam, N)
            concat_h_state = torch.cat(
                [
                    h_t_state.view(-1, beam_size, ndim),
                    h_u_t_state.view(-1, beam_size, ndim),
                    h_u_u_t_state.view(-1, beam_size, ndim),
                ],
                dim=1,
            )
            # (B, 3*beam, N)
            concat_c_state = torch.cat(
                [
                    c_t_state.view(-1, beam_size, ndim),
                    c_u_t_state.view(-1, beam_size, ndim),
                    c_u_u_t_state.view(-1, beam_size, ndim),
                ],
                dim=1,
            )
            out_states.append((concat_h_state.view(-1, ndim), concat_c_state.view(-1, ndim)))
    elif lstm_type == 'Pytorch':
        h_t_state, c_t_state = t_states
        nlayer = h_t_state.size(0)
        ndim = h_t_state.size(-1)
        h_u_t_state, c_u_t_state = u_t_states
        h_u_u_t_state, c_u_u_t_state = u_u_t_states
        # (nlayer, B, 3*beam, ndim)
        concat_h_state = torch.cat(
            [
                h_t_state.view(nlayer, -1, beam_size, ndim),
                h_u_t_state.view(nlayer, -1, beam_size, ndim),
                h_u_u_t_state.view(nlayer, -1, beam_size, ndim),
            ],
            dim=2,
        )
        # (nlayer, B, 3*beam, ndim)
        concat_c_state = torch.cat(
            [
                c_t_state.view(nlayer, -1, beam_size, ndim),
                c_u_t_state.view(nlayer, -1, beam_size, ndim),
                c_u_u_t_state.view(nlayer, -1, beam_size, ndim),
            ],
            dim=2,
        )
        out_states = (concat_h_state.view(nlayer, -1, ndim), concat_c_state.view(nlayer, -1, ndim))
    elif lstm_type == 'LimitedContext':
        ndim = t_states.size(-1)
        concat_state = torch.cat(
            [
                t_states.view(-1, beam_size, ndim),
                u_t_states.view(-1, beam_size, ndim),
                u_u_t_states.view(-1, beam_size, ndim),
            ],
            dim=1,
        )
        out_states = concat_state.view(-1, ndim)
    return out_states


def get_prev_emb(emb_weight, prev_tgt, prev_char_drop=0, unk_idx=0, fused=True):
    '''
    get prev embedding
    Args:
        emb_weight: [V, H]
        prev_tgt: [B, U]
    Return:
        prev_emb: [B, U, H]
    '''
    if fused:
        return fused_prev_char(
            emb_weight,
            prev_tgt,
            dropout_factor=prev_char_drop,
            unk_idx=unk_idx,
        )
    bsz, tgt_num = prev_tgt.size()
    if prev_char_drop > 0.0:
        with torch.no_grad():
            random_uniform_tensor = torch.empty(bsz, tgt_num).uniform_(0, prev_char_drop + 1).cuda()
            mask = (random_uniform_tensor < 1).long()
            prev_tgt = prev_tgt * mask + (1 - mask) * unk_idx
    return torch.embedding(emb_weight, prev_tgt)


class LSTMPredictor(nn.Module):
    '''LSTM predictor: composed by a embedding layer, several lstm layers,
    and a linear layer with LayerNorm
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.predictor_emb_size)
        self.lstm = IncrementalLSTM(
            args.predictor_emb_size,
            args.predictor_lstm_hidden_size,
            num_layers=args.predictor_lstm_layer_num,
            batch_first=True,
            bidirectional=False,
        )
        self.setup_proj_fc(args)

    def setup_proj_fc(self, args):
        '''setup proj fc.'''
        self.lstm_proj_fc = nn.Sequential(
            *[
                nn.Linear(args.predictor_lstm_hidden_size, args.jointer_hidden_size),
                nn.LayerNorm(args.jointer_hidden_size),
            ]
        )

    @property
    def state_size(self):
        '''state size.'''
        return self.lstm.num_layers * 2 * self.lstm.hidden_size

    def forward(self, prev_tgt, incremental_states=None, prev_char_drop=0, unk_idx=0, **kwargs):
        '''forward
        Args:
            prev_tgt: [B, U], previous target
            incremental_states: lstm states
        Return:
            predictor_out: [B, U, H], predictor output
            incremental_states: lstm final states
        '''
        return_embed = kwargs.get('return_embed', False)
        prev_emb = get_prev_emb(
            self.embed_tokens.weight,
            prev_tgt,
            prev_char_drop=prev_char_drop,
            unk_idx=unk_idx,
            fused=self.training,
        )
        if incremental_states is not None:
            lstm_out, incremental_states = self.lstm.forward_step(prev_emb, incremental_states)
        else:
            lstm_out, incremental_states = self.lstm(prev_emb)
        predictor_out = self.lstm_proj_fc(lstm_out)
        if return_embed:
            return prev_emb, predictor_out, incremental_states
        return predictor_out, incremental_states

    def forward_step(self, prev_tgt, incremental_states=None, **kwargs):
        '''froward by a token step'''
        if isinstance(incremental_states, torch.Tensor):
            assert incremental_states.shape[1] == self.state_size
            offset, hidden_size = 0, self.lstm.hidden_size
            new_states = []
            for _ in range(self.lstm.num_layers):
                h0 = incremental_states[:, offset : offset + hidden_size]
                offset += hidden_size
                c0 = incremental_states[:, offset : offset + hidden_size]
                offset += hidden_size
                new_states.append((h0, c0))
            incremental_states = new_states

        return_embed = kwargs.get('return_embed', False)
        prev_emb = self.embed_tokens(prev_tgt)
        lstm_out, incremental_states = self.lstm.forward_step(prev_emb, incremental_states)
        predictor_out = self.lstm_proj_fc(lstm_out)
        if return_embed:
            return (prev_emb, predictor_out), incremental_states
        return predictor_out, incremental_states

    @staticmethod
    def select_non_blank_states(incremental_states, incremental_states_next, mask):
        '''select_non_blank_states'''
        return lstm_select_non_blank_states(incremental_states, incremental_states_next, mask)

    @staticmethod
    def first_step_expand_state(states, beam_size):
        '''first_step_expand_state'''
        return lstm_first_step_expand_state(states, beam_size)

    @staticmethod
    def first_step_concate_state(init_states, expand_states, beam_size):
        '''first_step_concate_state'''
        return lstm_first_step_concate_state(init_states, expand_states, beam_size)

    @staticmethod
    def reorder_beam_states(states, beam_idx):
        '''reorder_beam_states'''
        return lstm_reorder_beam_states(states, beam_idx)

    @staticmethod
    def concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size):
        '''concate_t_ut_uut_states'''
        return lstm_concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size)

    @staticmethod
    def concate_t_ut_states(t_states, u_t_states, beam_size):
        '''concate_t_ut_states'''
        return lstm_concate_t_ut_states(t_states, u_t_states, beam_size)


class ReducedEmbeddingPredictor(nn.Module):
    '''ReducedEmbedding predictor: composed by a embedding layer,
    and a linear layer with LayerNorm
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.predictor_emb_size)
        self.predictor_emb_size = args.predictor_emb_size
        self.predictor_head = args.predictor_head
        self.position_embedding_trainable = args.position_embedding_trainable
        self.limited_context = args.limited_context
        self.pos_vec = torch.rand(
            [args.predictor_head, args.limited_context, args.predictor_emb_size],
            device='cuda',
            requires_grad=args.position_embedding_trainable,
        )

        self.reduced_embedding = IncrementalReducedEmbedding(
            args.predictor_emb_size, args.predictor_head, args.limited_context
        )

        self.reduced_embedding_proj_fc = nn.Sequential(
            *[
                nn.Linear(args.predictor_emb_size, args.jointer_hidden_size),
                nn.LayerNorm(args.jointer_hidden_size),
            ],
        )
        self.predictor_activation_fn = args.predictor_activation_fn
        if self.predictor_activation_fn is not None:
            self.activation_fn = get_activation_fn(args.predictor_activation_fn)

    @property
    def state_size(self):
        '''state size.'''
        return (self.limited_context - 2) * self.predictor_emb_size

    def forward(self, prev_tgt, incremental_states=None, prev_char_drop=0, unk_idx=0, **_kwargs):
        """forward"""
        prev_emb = get_prev_emb(
            self.embed_tokens.weight,
            prev_tgt,
            prev_char_drop=prev_char_drop,
            unk_idx=unk_idx,
            fused=self.training,
        )
        pos_emb = self.pos_vec.sum(dim=0)
        prev0_emb = self.embed_tokens(torch.tensor([0], device='cuda'))
        if incremental_states is not None:
            reduced_embedding_out, incremental_states = self.reduced_embedding.forward_step(
                prev_emb, pos_emb, incremental_states, prev0_emb
            )
        else:
            reduced_embedding_out, incremental_states = self.reduced_embedding(prev_emb, pos_emb)
        predictor_out = self.reduced_embedding_proj_fc(reduced_embedding_out)
        if self.predictor_activation_fn is not None:
            predictor_out = self.activation_fn(predictor_out)
        return predictor_out, incremental_states

    def forward_step(self, prev_tgt, incremental_states=None, **_kwargs):
        '''froward by a token step'''
        prev_emb = self.embed_tokens(prev_tgt)
        pos_emb = self.pos_vec.sum(dim=0)
        prev0_emb = self.embed_tokens(torch.tensor([0], device='cuda'))
        reduced_embedding_out, incremental_states = self.reduced_embedding.forward_step(
            prev_emb, pos_emb, incremental_states, prev0_emb
        )
        predictor_out = self.reduced_embedding_proj_fc(reduced_embedding_out)
        if self.predictor_activation_fn is not None:
            predictor_out = self.activation_fn(predictor_out)
        return predictor_out, incremental_states

    @staticmethod
    def select_non_blank_states(incremental_states, incremental_states_next, mask):
        '''select_non_blank_states'''
        return lstm_select_non_blank_states(
            incremental_states, incremental_states_next, mask, lstm_type='LimitedContext'
        )

    @staticmethod
    def first_step_expand_state(states, beam_size):
        '''first_step_expand_state'''
        return lstm_first_step_expand_state(states, beam_size, lstm_type='LimitedContext')

    @staticmethod
    def first_step_concate_state(init_states, expand_states, beam_size):
        '''first_step_concate_state'''
        return lstm_first_step_concate_state(
            init_states, expand_states, beam_size, lstm_type='LimitedContext'
        )

    @staticmethod
    def reorder_beam_states(states, beam_idx):
        '''reorder_beam_states'''
        return lstm_reorder_beam_states(states, beam_idx, lstm_type='LimitedContext')

    @staticmethod
    def concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size):
        '''concate_t_ut_uut_states'''
        return lstm_concate_t_ut_uut_states(
            t_states, u_t_states, u_u_t_states, beam_size, lstm_type='LimitedContext'
        )

    @staticmethod
    def concate_t_ut_states(t_states, u_t_states, beam_size):
        '''concate_t_ut_states'''
        return lstm_concate_t_ut_states(t_states, u_t_states, beam_size, lstm_type='LimitedContext')


class LimitedContextLSTMPredictor(nn.Module):
    '''LimitedContextLSTM predictor: composed by a embedding layer, several lstm layers,
    and a linear layer with LayerNorm
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.predictor_emb_size)
        self.limited_context = args.limited_context
        self.limited_context_padding = args.limited_context_padding
        self.limited_context_lstm = IncrementalLimitedContextLSTM(
            args.predictor_emb_size,
            args.predictor_lstm_hidden_size,
            args.predictor_lstm_layer_num,
            args.jointer_hidden_size,
            args.limited_context,
            args.limited_context_padding,
            batch_first=True,
            bidirectional=False,
        )

        self.limited_context_lstm_proj_fc = nn.Sequential(
            *[
                nn.Linear(args.predictor_lstm_hidden_size, args.jointer_hidden_size),
                nn.LayerNorm(args.jointer_hidden_size),
            ]
        )
        self.limited_context_padding = args.limited_context_padding

    def forward(self, prev_tgt, incremental_states=None, prev_char_drop=0, unk_idx=0, **_kwargs):
        """forward"""
        prev_emb = get_prev_emb(
            self.embed_tokens.weight,
            prev_tgt,
            prev_char_drop=prev_char_drop,
            unk_idx=unk_idx,
            fused=self.training,
        )
        prev0_emb = self.embed_tokens(torch.tensor([0], device='cuda'))

        if incremental_states is not None:
            (
                limited_context_lstm_out,
                incremental_states,
                index,
            ) = self.limited_context_lstm.forward_step(prev_emb, incremental_states, prev0_emb)
        else:
            limited_context_lstm_out, incremental_states, index = self.limited_context_lstm(
                prev_emb, prev0_emb
            )
        if self.limited_context_padding is not True:
            gather_index = (
                index.expand(
                    limited_context_lstm_out.size(1),
                    limited_context_lstm_out.size(2),
                    index.size(0),
                )
                .permute(2, 0, 1)
                .contiguous()
            )
            split_predictor_out = (
                torch.gather(limited_context_lstm_out, 1, gather_index)
                .permute(1, 0, 2)
                .contiguous()
            )[0][:][:]
            predictor_out = self.limited_context_lstm_proj_fc(split_predictor_out)
        else:
            assert False

        return predictor_out, incremental_states

    def forward_step(self, prev_tgt, incremental_states=None):
        '''froward by a token step'''
        prev_emb = self.embed_tokens(prev_tgt)
        prev0_emb = self.embed_tokens(torch.tensor([0], device='cuda'))
        (
            limited_context_lstm_out,
            incremental_states,
            index,
        ) = self.limited_context_lstm.forward_step(prev_emb, incremental_states, prev0_emb)

        if self.limited_context_padding is not True:
            gather_index = (
                index.expand(
                    limited_context_lstm_out.size(1),
                    limited_context_lstm_out.size(2),
                    index.size(0),
                )
                .permute(2, 0, 1)
                .contiguous()
            )
            split_predictor_out = (
                torch.gather(limited_context_lstm_out, 1, gather_index)
                .permute(1, 0, 2)
                .contiguous()
            )[0][:][:]
            predictor_out = self.limited_context_lstm_proj_fc(split_predictor_out)
        else:
            assert False

        return predictor_out, incremental_states

    def select_non_blank_states(self, incremental_states, incremental_states_next, mask):
        '''select_non_blank_states'''
        mask = mask.expand(mask.size(0), self.limited_context).unsqueeze(2)
        return lstm_select_non_blank_states(
            incremental_states, incremental_states_next, mask, lstm_type='LimitedContext'
        )

    @staticmethod
    def first_step_expand_state(states, beam_size):
        '''first_step_expand_state'''
        return lstm_first_step_expand_state(states, beam_size, lstm_type='LimitedContext')

    @staticmethod
    def first_step_concate_state(init_states, expand_states, beam_size):
        '''first_step_concate_state'''
        return lstm_first_step_concate_state(
            init_states, expand_states, beam_size, lstm_type='LimitedContext'
        )

    @staticmethod
    def reorder_beam_states(states, beam_idx):
        '''reorder_beam_states'''
        return lstm_reorder_beam_states(states, beam_idx, lstm_type='LimitedContext')

    @staticmethod
    def concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size):
        '''concate_t_ut_uut_states'''
        return lstm_concate_t_ut_uut_states(
            t_states, u_t_states, u_u_t_states, beam_size, lstm_type='LimitedContext'
        )

    @staticmethod
    def concate_t_ut_states(t_states, u_t_states, beam_size):
        '''concate_t_ut_states'''
        return lstm_concate_t_ut_states(t_states, u_t_states, beam_size, lstm_type='LimitedContext')


class LSTMReLUPredictor(LSTMPredictor):
    '''LSTMReLUPredictor
    The same as LSTMPredictor, with LayerNorm replaced by ReLU
    '''

    def setup_proj_fc(self, args):
        '''setup proj fc.'''
        self.lstm_proj_fc = nn.Sequential(
            *[
                nn.Linear(args.predictor_lstm_hidden_size, args.jointer_hidden_size),
                nn.LeakyReLU(0.2),
            ]
        )


class PytorchLSTMPredictor(LSTMPredictor):
    '''PytorchLSTMPredictor
    same as LSTMPredictor except that uses
    Pytorch LSTM instead of Incremental LSTM
    '''

    def __init__(self, args):
        super().__init__(args)
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.predictor_emb_size)

        self.lstm = nn.LSTM(
            args.predictor_emb_size,
            args.predictor_lstm_hidden_size,
            num_layers=args.predictor_lstm_layer_num,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm_proj_fc = nn.Sequential(
            *[
                nn.Linear(args.predictor_lstm_hidden_size, args.jointer_hidden_size),
                nn.LayerNorm(args.jointer_hidden_size),
            ]
        )

    def forward(self, prev_tgt, prev_states=None, prev_char_drop=0, unk_idx=0, **_kwargs):
        prev_emb = get_prev_emb(
            self.embed_tokens.weight,
            prev_tgt,
            prev_char_drop=prev_char_drop,
            unk_idx=unk_idx,
            fused=self.training,
        )
        if prev_states is not None:
            lstm_out, incremental_states = self.lstm(prev_emb, prev_states)
        else:
            lstm_out, incremental_states = self.lstm(prev_emb)
        predictor_out = self.lstm_proj_fc(lstm_out)
        return predictor_out, incremental_states

    def forward_step(self, prev_tgt, prev_states=None, **_kwargs):
        predictor_out, incremental_states = self.forward(prev_tgt.unsqueeze(1), prev_states)
        return predictor_out.squeeze(1), incremental_states

    @staticmethod
    def select_non_blank_states(incremental_states, incremental_states_next, mask):
        '''select_non_blank_states'''
        return lstm_select_non_blank_states(
            incremental_states, incremental_states_next, mask, lstm_type='Pytorch'
        )

    @staticmethod
    def first_step_expand_state(states, beam_size):
        '''first_step_expand_state'''
        return lstm_first_step_expand_state(states, beam_size, lstm_type='Pytorch')

    @staticmethod
    def first_step_concate_state(init_states, expand_states, beam_size):
        '''first_step_concate_state'''
        return lstm_first_step_concate_state(
            init_states, expand_states, beam_size, lstm_type='Pytorch'
        )

    @staticmethod
    def reorder_beam_states(states, beam_idx):
        '''reorder_beam_states'''
        return lstm_reorder_beam_states(states, beam_idx, lstm_type='Pytorch')

    @staticmethod
    def concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size):
        '''concate_t_ut_uut_states'''
        return lstm_concate_t_ut_uut_states(
            t_states, u_t_states, u_u_t_states, beam_size, lstm_type='Pytorch'
        )

    @staticmethod
    def concate_t_ut_states(t_states, u_t_states, beam_size):
        '''concate_t_ut_states'''
        return lstm_concate_t_ut_states(t_states, u_t_states, beam_size, lstm_type='Pytorch')


class LSTMProjPredictor(nn.Module):
    '''LSTMP predictor'''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, args.predictor_emb_size)
        self.lstm_list = nn.ModuleList()
        self.lstm_proj_list = nn.ModuleList()
        self.lstm_layer_num = args.predictor_lstm_layer_num
        for i in range(args.predictor_lstm_layer_num):
            if i == 0:
                lstm_input_size = args.predictor_emb_size
            else:
                lstm_input_size = args.jointer_hidden_size
            self.lstm_list.append(
                IncrementalLSTM(
                    lstm_input_size,
                    args.predictor_lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )
            )
            self.lstm_proj_list.append(
                nn.Sequential(
                    *[
                        nn.Linear(args.predictor_lstm_hidden_size, args.jointer_hidden_size),
                        nn.LayerNorm(args.jointer_hidden_size),
                    ]
                )
            )

    def forward(self, prev_tgt, incremental_states=None, prev_char_drop=0, unk_idx=0, **_kwargs):
        '''forward: the same as that in LSTMPredictor'''
        prev_emb = get_prev_emb(
            self.embed_tokens.weight,
            prev_tgt,
            prev_char_drop=prev_char_drop,
            unk_idx=unk_idx,
            fused=self.training,
        )
        new_lstm_states = []
        for i in range(self.lstm_layer_num):
            if i == 0:
                lstm_input = prev_emb
            else:
                lstm_input = lstm_proj_out
            if incremental_states is not None:
                lstm_out, incremental_state = self.lstm_list[i].forward_step(
                    lstm_input, incremental_states[i]
                )
            else:
                lstm_out, incremental_state = self.lstm_list[i](lstm_input)
            lstm_proj_out = self.lstm_proj_list[i](lstm_out)
            new_lstm_states.append(incremental_state)
        return lstm_proj_out, new_lstm_states

    def forward_step(self, prev_tgt, incremental_states=None, **_kwargs):
        '''froward by a token step'''
        prev_emb = self.embed_tokens(prev_tgt)
        new_lstm_states = []
        for i in range(self.lstm_layer_num):
            if i == 0:
                lstm_input = prev_emb
            else:
                lstm_input = lstm_proj_out
            if incremental_states is None:
                lstm_out, incremental_state = self.lstm_list[i].forward_step(lstm_input, None)
            else:
                lstm_out, incremental_state = self.lstm_list[i].forward_step(
                    lstm_input, incremental_states[i]
                )
            lstm_proj_out = self.lstm_proj_list[i](lstm_out)
            new_lstm_states.append(incremental_state)
        return lstm_proj_out, new_lstm_states

    @staticmethod
    def select_non_blank_states(incremental_states, incremental_states_next, mask):
        '''select_non_blank_states'''
        return lstm_select_non_blank_states(incremental_states, incremental_states_next, mask)

    @staticmethod
    def first_step_expand_state(states, beam_size):
        '''first_step_expand_state'''
        return lstm_first_step_expand_state(states, beam_size)

    @staticmethod
    def first_step_concate_state(init_states, expand_states, beam_size):
        '''first_step_concate_state'''
        return lstm_first_step_concate_state(init_states, expand_states, beam_size)

    @staticmethod
    def reorder_beam_states(states, beam_idx):
        '''reorder_beam_states'''
        return lstm_reorder_beam_states(states, beam_idx)

    @staticmethod
    def concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size):
        '''concate_t_ut_uut_states'''
        return lstm_concate_t_ut_uut_states(t_states, u_t_states, u_u_t_states, beam_size)

    @staticmethod
    def concate_t_ut_states(t_states, u_t_states, beam_size):
        '''concate_t_ut_states'''
        return lstm_concate_t_ut_states(t_states, u_t_states, beam_size)


class TransformerXLPredictor(nn.Module):
    """TransformerXLPredictor
    https://github.com/kimiyoung/transformer-xl/
    """

    # pylint:disable=unnecessary-list-index-lookup

    def __init__(self, args):
        '''constructor'''
        super(__class__, self).__init__()
        self.args = args

        self.n_layer = args.predictor_n_layer

        self.d_embed = args.predictor_d_embed
        self.d_model = args.predictor_d_model
        self.n_head = args.predictor_n_head
        self.d_head = args.predictor_d_head
        self.d_inner = args.predictor_d_inner
        self.dropout = args.predictor_dropout
        self.dropatt = args.predictor_dropatt

        self.pre_lnorm = args.predictor_pre_lnorm

        self.ext_len = args.predictor_ext_len
        self.mem_len = args.predictor_mem_len

        self.activation_fn = args.predictor_activation_fn
        self.swish_beta = (
            args.get("predictor_swish_beta", 1.0) if self.activation_fn == 'swish' else None
        )
        self.relu_inplace = (
            args.get("predictor_relu_inplace", False) if self.activation_fn == 'relu' else None
        )

        self.embed_tokens = nn.Embedding(args.tgt_vocab_size, self.d_embed)

        self.drop = nn.Dropout(self.dropout)

        self.layers = nn.ModuleList()

        for _ in range(self.n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    self.n_head,
                    self.d_model,
                    self.d_head,
                    self.d_inner,
                    self.dropout,
                    dropatt=self.dropatt,
                    pre_lnorm=self.pre_lnorm,
                    activation_fn=self.activation_fn,
                    swish_beta=self.swish_beta,
                    relu_inplace=self.relu_inplace,
                )
            )

        self._create_params()

        self.transformer_proj_fc = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.Linear(self.d_model, args.jointer_hidden_size)
        )
        self.init_params()

    def init_params(self):
        '''init params'''
        # pylint:disable=too-many-branches

        def init_weight(weight):
            if self.args.init == 'uniform':
                nn.init.uniform_(weight, -self.args.init_range, self.args.init_range)
            elif self.args.init == 'normal':
                nn.init.normal_(weight, 0.0, self.args.init_std)

        def init_bias(bias):
            nn.init.constant_(bias, 0.0)

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init_weight(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    init_bias(m.bias)
            elif classname.find('Embedding') != -1:
                if hasattr(m, 'weight'):
                    init_weight(m.weight)
            elif classname.find('LayerNorm') != -1:
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, 1.0, self.args.init_std)
                if hasattr(m, 'bias') and m.bias is not None:
                    init_bias(m.bias)
            elif classname.find('TransformerXLPredictor') != -1:
                if hasattr(m, 'r_emb'):
                    init_weight(m.r_emb)
                if hasattr(m, 'r_w_bias'):
                    init_weight(m.r_w_bias)
                if hasattr(m, 'r_r_bias'):
                    init_weight(m.r_r_bias)
                if hasattr(m, 'r_bias'):
                    init_bias(m.r_bias)

        self.apply(weights_init)

    def _create_params(self):
        '''create params'''
        self.pos_emb = SimpleSinusoidalPositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, ext_len, mem_len):
        '''reset length'''
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, bsz):
        '''init mems'''
        if self.mem_len > 0:
            mems = []
            mems_mask = []
            param = next(self.parameters())
            for _ in range(self.n_layer + 1):
                mem = torch.zeros(
                    self.mem_len,
                    bsz,
                    self.args.predictor_d_model,
                    dtype=param.dtype,
                    device=param.device,
                )
                mem_mask = torch.ones(self.mem_len, bsz, dtype=torch.uint8, device=param.device)
                mems.append(mem)
                mems_mask.append(mem_mask)

            return mems, mems_mask
        return None, None

    def _update_mems(self, hids, mems, mems_mask, qlen, mlen, bsz):
        '''update mems'''
        # does not deal with None
        if mems is None:
            return None, None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            new_mems_mask = []
            end_idx = mlen + max(0, qlen - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i, _ in enumerate(hids):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
                all_zeros = torch.zeros(qlen, bsz, dtype=torch.uint8, device=cat.device)
                cat_mask = torch.cat([mems_mask[i], all_zeros], dim=0)
                new_mems_mask.append(cat_mask[beg_idx:end_idx].detach())

        return new_mems, new_mems_mask

    def forward(self, prev_tgt, incremental_states=None, prev_char_drop=0, unk_idx=0, **_kwargs):
        '''forward'''
        bsz, qlen = prev_tgt.size()

        if not incremental_states:
            mems, mems_mask = self.init_mems(bsz)
        else:
            mems, mems_mask = incremental_states

        word_emb = get_prev_emb(
            self.embed_tokens.weight,
            prev_tgt,
            prev_char_drop=prev_char_drop,
            unk_idx=unk_idx,
            fused=self.training,
        )
        word_emb = word_emb.permute(1, 0, 2)  # [U, B, H]

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        all_ones = word_emb.new_ones(qlen, klen)
        mask_len = klen - self.mem_len
        mask_shift_len = qlen - mask_len if mask_len > 0 else qlen
        dec_attn_mask = (
            torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -1 - mask_shift_len)
        ).byte()[:, :, None]
        dec_attn_mask = dec_attn_mask.repeat(1, 1, bsz)

        hids = []

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)

        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            if mems_mask is not None:
                mems_mask_i = mems_mask[i]
                dec_attn_mask[:, :mlen, :] = mems_mask_i.unsqueeze(0).repeat(qlen, 1, 1)
            core_out = layer(
                core_out,
                pos_emb,
                self.r_w_bias,
                self.r_r_bias,
                dec_attn_mask=dec_attn_mask,
                mems=mems_i,
            )
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems, new_mems_mask = self._update_mems(hids, mems, mems_mask, qlen, mlen, bsz)

        core_out = core_out.transpose(0, 1)

        predictor_out = self.transformer_proj_fc(core_out)
        return predictor_out, (new_mems, new_mems_mask)

    def forward_step(self, prev_tgt, incremental_states=None, **_kwargs):
        '''forward step'''
        predictor_out, new_incremental_states = self.forward(
            prev_tgt.unsqueeze(1), incremental_states
        )
        return predictor_out.squeeze(1), new_incremental_states

    def select_non_blank_states(self, incremental_states, incremental_states_next, mask):
        '''select non_blank states'''
        incremental_states, incremental_states_next = self.align_states(
            (incremental_states, incremental_states_next)
        )
        mems, mems_mask = incremental_states
        mems_next, mems_mask_next = incremental_states_next
        new_mems, new_mems_mask = [], []
        for i, _ in enumerate(mems):
            mem_i, mem_mask_i = mems[i], mems_mask[i]
            mem_next_i, mem_mask_next_i = mems_next[i], mems_mask_next[i]
            new_mem_i = torch.where(mask.unsqueeze(0), mem_i, mem_next_i)
            new_mems.append(new_mem_i)
            new_mem_mask_i = torch.where(mask.unsqueeze(0).squeeze(-1), mem_mask_i, mem_mask_next_i)
            new_mems_mask.append(new_mem_mask_i)
        return (new_mems, new_mems_mask)

    @staticmethod
    def first_step_expand_state(states, beam_size):
        '''first step expand state'''
        mems, mems_mask = states
        expand_mems, expand_mems_mask = [], []
        mlen, _, ndim = mems[0].size()
        for i, _ in enumerate(mems):
            mem_i, mem_mask_i = mems[i], mems_mask[i]
            expand_mem_i = (
                mem_i.unsqueeze(2).repeat(1, 1, beam_size, 1).contiguous().view(mlen, -1, ndim)
            )
            expand_mem_mask_i = (
                mem_mask_i.unsqueeze(2).repeat(1, 1, beam_size).contiguous().view(mlen, -1)
            )
            expand_mems.append(expand_mem_i)
            expand_mems_mask.append(expand_mem_mask_i)
        return (expand_mems, expand_mems_mask)

    @staticmethod
    def first_step_concate_state(init_states, expand_states, beam_size):
        '''first step concate state'''
        mems, mems_mask = init_states
        expand_mems, expand_mems_mask = expand_states
        concat_mems, concat_mems_mask = [], []
        mlen, bsz, ndim = mems[0].size()
        for i, _ in enumerate(mems):
            mem_i, mem_mask_i = mems[i], mems_mask[i]
            expand_mem_i, expand_mem_mask_i = expand_mems[i], expand_mems_mask[i]
            concat_mem_i = (
                torch.cat([mem_i.unsqueeze(2), expand_mem_i.view(mlen, bsz, -1, ndim)], dim=2)
                .contiguous()
                .view(mlen, bsz * beam_size, ndim)
            )
            concat_mem_mask_i = (
                torch.cat([mem_mask_i.unsqueeze(2), expand_mem_mask_i.view(mlen, bsz, -1)], dim=2)
                .contiguous()
                .view(mlen, bsz * beam_size)
            )
            concat_mems.append(concat_mem_i)
            concat_mems_mask.append(concat_mem_mask_i)
        return (concat_mems, concat_mems_mask)

    @staticmethod
    def reorder_beam_states(states, beam_idx):
        '''reorder beam states'''
        mems, mems_mask = states
        reorder_mems, reorder_mems_mask = [], []
        mlen, _, ndim = mems[0].size()
        bsz, _ = beam_idx.size()
        for i, _ in enumerate(mems):
            mem_i, mem_mask_i = mems[i], mems_mask[i]

            reorder_mem_i = (
                mem_i.view(mlen, bsz, -1, ndim)
                .gather(dim=2, index=beam_idx.unsqueeze(0).unsqueeze(-1).repeat(mlen, 1, 1, ndim))
                .contiguous()
                .view(mlen, -1, ndim)
            )

            reorder_mem_mask_i = (
                mem_mask_i.view(mlen, bsz, -1)
                .gather(dim=2, index=beam_idx.unsqueeze(0).repeat(mlen, 1, 1))
                .contiguous()
                .view(mlen, -1)
            )

            reorder_mems.append(reorder_mem_i)
            reorder_mems_mask.append(reorder_mem_mask_i)
        return (reorder_mems, reorder_mems_mask)

    def concate_t_ut_uut_states(self, t_states, u_t_states, u_u_t_states, beam_size):
        '''concate_t_ut_uut_states'''
        t_states, u_t_states, u_u_t_states = self.align_states((t_states, u_t_states, u_u_t_states))
        t_mems, t_mems_mask = t_states
        u_t_mems, u_t_mems_mask = u_t_states
        u_u_t_mems, u_u_t_mems_mask = u_u_t_states

        concat_mems, concat_mems_mask = [], []

        mlen, _, ndim = t_mems[0].size()
        for i, _ in enumerate(t_mems):
            t_mem_i, t_mem_mask_i = t_mems[i], t_mems_mask[i]
            u_t_mem_i, u_t_mem_mask_i = u_t_mems[i], u_t_mems_mask[i]
            u_u_t_mem_i, u_u_t_mem_mask_i = u_u_t_mems[i], u_u_t_mems_mask[i]

            concat_mem_i = (
                torch.cat(
                    [
                        t_mem_i.view(mlen, -1, beam_size, ndim),
                        u_t_mem_i.view(mlen, -1, beam_size, ndim),
                        u_u_t_mem_i.view(mlen, -1, beam_size, ndim),
                    ],
                    dim=2,
                )
                .contiguous()
                .view(mlen, -1, ndim)
            )

            concat_mem_mask_i = (
                torch.cat(
                    [
                        t_mem_mask_i.view(mlen, -1, beam_size),
                        u_t_mem_mask_i.view(mlen, -1, beam_size),
                        u_u_t_mem_mask_i.view(mlen, -1, beam_size),
                    ],
                    dim=2,
                )
                .contiguous()
                .view(mlen, -1)
            )

            concat_mems.append(concat_mem_i)
            concat_mems_mask.append(concat_mem_mask_i)
        return (concat_mems, concat_mems_mask)

    @staticmethod
    def align_states(args):
        '''align states'''
        mlen, bsz, ndim = args[-1][0][0].size()
        for idx, _ in enumerate(args):
            cur_mems, cur_mems_mask = args[idx]
            cur_len = cur_mems[0].size(0)
            padding = cur_mems[0].data.new(mlen - cur_len, bsz, ndim).zero_()
            mask = cur_mems_mask[0].data.new(mlen - cur_len, bsz).fill_(1)
            for i, _ in enumerate(cur_mems):
                cur_mem_i, cur_mem_mask_i = cur_mems[i], cur_mems_mask[i]
                args[idx][0][i] = torch.cat((padding, cur_mem_i), dim=0)
                args[idx][1][i] = torch.cat((mask, cur_mem_mask_i), dim=0)
        return args

    def concate_t_ut_states(self, t_states, u_t_states, beam_size):
        '''concate_t_ut_states'''
        t_states, u_t_states = self.align_states((t_states, u_t_states))
        t_mems, t_mems_mask = t_states
        u_t_mems, u_t_mems_mask = u_t_states
        concat_mems, concat_mems_mask = [], []
        mlen, _, ndim = t_mems[0].size()
        for t_mem_i, t_mem_mask_i, u_t_mem_i, u_t_mem_mask_i in zip(
            t_mems, t_mems_mask, u_t_mems, u_t_mems_mask
        ):
            concat_mem_i = (
                torch.cat(
                    [
                        t_mem_i.view(mlen, -1, beam_size, ndim),
                        u_t_mem_i.view(mlen, -1, beam_size, ndim),
                    ],
                    dim=2,
                )
                .contiguous()
                .view(mlen, -1, ndim)
            )
            concat_mem_mask_i = (
                torch.cat(
                    [
                        t_mem_mask_i.view(mlen, -1, beam_size),
                        u_t_mem_mask_i.view(mlen, -1, beam_size),
                    ],
                    dim=2,
                )
                .contiguous()
                .view(mlen, -1)
            )

            concat_mems.append(concat_mem_i)
            concat_mems_mask.append(concat_mem_mask_i)
        return (concat_mems, concat_mems_mask)


class ContextAwareLSTMPredictor(LSTMReLUPredictor):
    '''Predictor network of Context-aware RNN-T'''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__(args)
        self.num_layers = args.predictor_lstm_layer_num
        self.hidden_size = args.predictor_lstm_hidden_size
        # concat the prev_embd with lstm-state in Predictor to build the attention query
        self.query_with_prev_embd = args.get('context_query_with_prev_embd', False)
        self.context_off = args.get('context_off', False)

    def forward(self, prev_tgt, incremental_states=None, **kwargs):
        '''forward
        Args:
            prev_tgt: [B, U], previous target
            incremental_states: lstm states
        Return:
            predictor_out: [B, U, H], predictor output
            incremental_states: lstm final states
        '''
        lstm_out_list = []
        for i in range(prev_tgt.shape[1]):
            step_out, incremental_states = self.forward_step(
                prev_tgt[:, i : i + 1].view(-1), incremental_states, **kwargs
            )
            lstm_out_list.append(step_out)
        predictor_out = torch.stack(lstm_out_list, dim=1)
        return predictor_out, incremental_states

    def forward_step(self, prev_tgt, incremental_states=None, **kwargs):
        '''forward
        Args:
            prev_tgt: [B,], previous target
            incremental_states: lstm states
        Return:
            predictor_out: [B, H], predictor output
            incremental_states: lstm step states
        '''
        context_embd = kwargs.get('context_embd')
        context_mask = kwargs.get('context_mask')
        context_encoder = kwargs.get('context_encoder')

        prev_embd = self.embed_tokens(prev_tgt)
        if incremental_states is None:
            incremental_states = []
            for _ in range(self.num_layers):
                zeros = torch.zeros(
                    prev_embd.size(0),
                    self.hidden_size,
                    dtype=prev_embd.dtype,
                    device=prev_embd.device,
                )
                incremental_states.append((zeros, zeros))

        if not self.context_off and context_embd is not None:
            if self.query_with_prev_embd:
                attn_query = torch.cat((incremental_states[-1][0], prev_embd), dim=1).unsqueeze(1)
            else:
                attn_query = incremental_states[-1][0].unsqueeze(1)
            context_out, _ = context_encoder.forward(attn_query, context_embd, context_mask)
            prev_embd = prev_embd + context_out.squeeze(1)

        lstm_out, incremental_states = self.lstm.forward_step(prev_embd, incremental_states)
        predictor_out = self.lstm_proj_fc(lstm_out)
        return predictor_out, incremental_states
