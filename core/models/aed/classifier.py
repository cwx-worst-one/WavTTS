""" Base classifier model """
import torch
from torch import nn
from core.models.layers.multi_head_attn import MultiheadAttention


class EmotionSingleLSTMHead(nn.Module):
    """Head for sequence output classification task."""

    def __init__(self, args, emotion_num, final_dropout=None, encoder_embed_dim=None):
        super().__init__()
        final_dropout = args.final_dropout if final_dropout is None else final_dropout
        encoder_embed_dim = (
            args.encoder_embed_dim if encoder_embed_dim is None else encoder_embed_dim
        )
        hidden_layer_dim = 64
        self.final_dropout = nn.Dropout(final_dropout)
        self.pred_fc = nn.Linear(hidden_layer_dim * 2, emotion_num)
        nn.init.xavier_uniform_(self.pred_fc.weight)
        nn.init.constant_(self.pred_fc.bias, 0.0)
        self.lstm1 = nn.LSTM(
            encoder_embed_dim, hidden_layer_dim, num_layers=1, batch_first=True, bidirectional=True
        )

    def forward(self, encoder_output):
        """forward"""
        encoder_output, (hidden, _) = self.lstm1(encoder_output)
        encoder_output = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        encoder_output = self.final_dropout(encoder_output)
        encoder_output = self.pred_fc(encoder_output)
        return encoder_output


class EmotionSeqOutHead(nn.Module):
    """Head for sequence output classification task."""

    def __init__(self, args, tgt_size, final_dropout=None, encoder_embed_dim=None):
        super().__init__()
        final_dropout = args.final_dropout if final_dropout is None else final_dropout
        encoder_embed_dim = (
            args.encoder_embed_dim if encoder_embed_dim is None else encoder_embed_dim
        )
        self.final_dropout = nn.Dropout(final_dropout)
        self.pred_fc = nn.Linear(encoder_embed_dim, tgt_size)
        nn.init.xavier_uniform_(self.pred_fc.weight)
        nn.init.constant_(self.pred_fc.bias, 0.0)

    def forward(self, encoder_output):
        """forward"""
        encoder_output = self.final_dropout(encoder_output)
        return self.pred_fc(encoder_output)


class EmotionMuiltmodalSeqOutHead(nn.Module):
    """Head for sequence output classification task."""

    def __init__(
        self,
        args,
        tgt_size,
        final_dropout=None,
        t_emotion_dim=768,
        a_emotion_dim=1024,
    ):
        super().__init__()
        final_dropout = args.final_dropout if final_dropout is None else final_dropout
        self.final_dropout = nn.Dropout(final_dropout)
        self.pred_fc = nn.Linear(a_emotion_dim, tgt_size)
        nn.init.xavier_uniform_(self.pred_fc.weight)
        nn.init.constant_(self.pred_fc.bias, 0.0)
        num_heads = args.get('attn_heads', 4)
        embed_dim = a_emotion_dim
        kdim = vdim = t_emotion_dim
        dropout_rate = args.get('context_dropout', 0.1)
        self.context_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout_rate,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
        )

    def forward(self, a_emotion, t_emotion, t_emotion_mask):
        """forward"""
        context_out, _ = self.context_attn(
            a_emotion.permute(1, 0, 2),
            t_emotion.permute(1, 0, 2),
            t_emotion.permute(1, 0, 2),
            (1 - t_emotion_mask) > 0,
        )
        context_out = context_out.permute(1, 0, 2)
        encoder_output = self.final_dropout(context_out)
        return self.pred_fc(encoder_output)


class EmotionSingleOutHead(nn.Module):
    """Head for single output classification task."""

    def __init__(
        self, args, emotion_num, regression=False, final_dropout=None, encoder_embed_dim=None
    ):
        super().__init__()
        self.args = args
        final_dropout = args.final_dropout if final_dropout is None else final_dropout
        encoder_embed_dim = (
            args.encoder_embed_dim if encoder_embed_dim is None else encoder_embed_dim
        )
        self.final_dropout = nn.Dropout(final_dropout)
        self.projector = nn.Linear(encoder_embed_dim, encoder_embed_dim)
        if regression:
            self.classifier = nn.Linear(encoder_embed_dim, 1)
        else:
            self.classifier = nn.Linear(encoder_embed_dim, emotion_num)
        nn.init.xavier_uniform_(self.projector.weight)
        nn.init.constant_(self.projector.bias, 0.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, encoder_output, padding_mask, hidden_states=None):
        """forward"""
        if hidden_states is not None:  # use_weighted_layer_sum
            hidden_states = [hidden_state.transpose(0, 1) for hidden_state in hidden_states]
            hidden_states = torch.stack(hidden_states, dim=1)
            encoder_output = hidden_states.mean(dim=1)
        if padding_mask is None:
            encoder_output = encoder_output.mean(dim=1)
        else:
            padding_mask = 1 - padding_mask.long()
            encoder_output = encoder_output * padding_mask.view(
                padding_mask.shape[0], padding_mask.shape[1], 1
            )
            encoder_output = encoder_output.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        encoder_output = self.final_dropout(encoder_output)
        encoder_output = self.projector(encoder_output)
        encoder_output = torch.tanh(encoder_output)
        encoder_output = self.final_dropout(encoder_output)
        return self.classifier(encoder_output)
