''' Context Encoder '''
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, BertModel as BertModel_huggingface
from core.models.layers.multi_head_attn import MultiheadAttention


class ContextEncoder(nn.Module):
    '''
    Context Module with attention
    attend to query (acoustic_feats or predictor states) and context inputs.
    '''

    def __init__(self, args):
        '''init
        Args:
            args: solution config
        '''
        super().__init__()
        self.update_steps = 0
        dropout_rate = args.get('context_dropout', 0)
        num_heads = args.get('context_attn_heads', 1)
        embed_dim = args.context_attn_qdim
        kdim = vdim = args.context_attn_kvdim

        self.context_encoder_fixed = args.get('context_encoder_fixed', False)
        self.context_encoder_fixed_steps = args.get('context_encoder_fixed_steps', 0)
        self.context_aware_method = args.get('context_aware_method', None)
        # concat the prev_embd with lstm-state in Predictor to build the attention query
        self.query_with_prev_embd = args.get('context_query_with_prev_embd', False)
        self.context_global_dropout = args.get('context_global_dropout', 0.0)
        self.context_encoder_type = args.get('context_encoder_type', 'LSTM')  # LSTM, BERT
        self.context_joint_training = args.get('context_joint_training', False)

        self.context_doc_emded_type = args.get('context_doc_emded_type', None)
        self.context_hidden_size = args.get('context_hidden_size', 768)

        self.context_cross_attn = args.get('context_cross_attn', False)

        if self.context_aware_method == 'CAP' and self.query_with_prev_embd:
            assert (
                args.context_attn_qdim == args.predictor_lstm_hidden_size + args.predictor_emb_size
            )

        encoder_config = args.get('context_encoder_cfg_local')
        if self.context_encoder_type == 'BERT':
            assert encoder_config is not None

            config = AutoConfig.from_pretrained(encoder_config)
            self.text_encoder = BertModel_huggingface(config=config)
        else:
            self.text_encoder = nn.Sequential(
                OrderedDict(
                    [
                        ('embed_tokens', nn.Embedding(args.tgt_vocab_size, args.context_embd_size)),
                        (
                            'lstm',
                            nn.LSTM(
                                args.context_embd_size,
                                args.context_lstm_hidden_size,
                                num_layers=args.context_lstm_layer_num,
                                batch_first=True,
                                bidirectional=args.get('context_bidirectional', False),
                            ),
                        ),
                    ]
                )
            )
        if self.context_doc_emded_type == "doc_rnn":
            hidden_size = self.context_hidden_size
            self.sentence_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        if self.context_doc_emded_type == "weight_sum_of_selfattn":
            self.ffn = nn.Linear(self.context_hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_rate)

        if self.context_cross_attn:
            kdim = embed_dim
            vdim = embed_dim
            self.cross_attn_layer = args.get('context_cross_attn_layer', 1)

            self.context_cross_ffn_pre = nn.Linear(self.context_hidden_size, embed_dim, bias=True)
            self.cross_attn = MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=kdim,
                vdim=vdim,
                dropout=dropout_rate,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
            )
            self.cross_self_attn = MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kdim=kdim,
                vdim=vdim,
                dropout=dropout_rate,
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
            )

            self.context_cross_ffn = nn.Linear(embed_dim, embed_dim, bias=True)

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
        self.proj_fc = nn.Linear(embed_dim, args.context_out_dim)

    def cross_attn_forward(self, context_embd, au_embd, layer):
        '''cross attention module
        Args:
            context_embd: context embeddings
            au_embd:  audio embeddings
            layer: cross attention layers
        Return:
            output: context embeddings
        '''
        context_embd = self.context_cross_ffn_pre(context_embd)
        for _ in range(layer):
            x = context_embd
            attn_out, _ = self.cross_attn(
                context_embd.permute(1, 0, 2),
                au_embd.permute(1, 0, 2),
                au_embd.permute(1, 0, 2),
            )
            attn_out = x + attn_out.permute(1, 0, 2)
            x = attn_out
            attn_out, _ = self.cross_self_attn(
                attn_out.permute(1, 0, 2),
                attn_out.permute(1, 0, 2),
                attn_out.permute(1, 0, 2),
            )
            attn_out = x + attn_out.permute(1, 0, 2)
            x = attn_out
            attn_out = self.context_cross_ffn(attn_out)
            context_embd = attn_out + x
        return context_embd

    def get_context_embd(self, context, context_mask=None):
        '''compute context embeddings
        Args:
            context: [B, U], context words
            context_mask: [B, U], context mask
        Return:
            output: [B, U, H], context embeddings
        '''
        if self.training:
            self.update_steps += 1
        bsz = context.shape[0]
        ori_context_dim = context.dim()
        if context.dim() == 3:
            context = context.reshape(-1, context.shape[2])
            context_mask = context_mask.reshape(-1, context_mask.shape[2])

        if self.context_encoder_fixed or self.update_steps < self.context_encoder_fixed_steps:
            with torch.no_grad():
                if self.context_encoder_type == 'BERT':
                    bert_input = {'input_ids': context, 'attention_mask': context_mask}
                    outputs = self.text_encoder(**bert_input)
                    context_embd = outputs['last_hidden_state']
                else:
                    context_embd, _ = self.text_encoder(context)
        else:
            with torch.enable_grad():
                if self.context_encoder_type == 'BERT':
                    bert_input = {'input_ids': context, 'attention_mask': context_mask}
                    outputs = self.text_encoder(**bert_input)
                    context_embd = outputs['last_hidden_state']
                else:
                    context_embd, _ = self.text_encoder(context)
        context_embd = self.dropout(context_embd)
        if context_mask is not None:
            context_embd = context_embd * context_mask.unsqueeze(2)
        if self.context_doc_emded_type is not None and ori_context_dim == 3:
            context_embd = context_embd.reshape(bsz, -1, context_embd.shape[2])
        return context_embd

    def dropout_global_context(self, context_out):
        '''discard the context by the dropout prob'''
        bsz = context_out.shape[0]
        if self.training and self.context_global_dropout > 0:
            mask = torch.ones(bsz, dtype=torch.float, device=context_out.device)
            mask = (F.dropout(mask, self.context_global_dropout) > 0).to(torch.float)
            context_out = context_out * mask.reshape(bsz, 1, 1)
        return context_out

    def forward(self, query, context, context_mask=None):
        '''forward
        Args:
            query: [B, T, H], the query of attention
            context: [B, U, D], context words indices or context embeddings
        Return:
            context_out: [B, T, H], context vector
            attn_weights: [B, T, U], attention weights
        '''

        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = (1.0 - context_mask) > 0

        if self.context_doc_emded_type is None:  # dialog context
            if context.dim() == 2:
                context_embd = self.get_context_embd(context, context_mask)
            elif context.dim() == 3:
                context_embd = context
            else:
                raise ValueError('context dim is wrong')

        else:  # document context
            key_padding_mask = None
            context_embd = self.get_context_embd(context, context_mask)
            bsz = context.shape[0]
            sent_num = context.shape[1]

            if self.context_doc_emded_type == "word_average":
                context_embd = context_embd.reshape(bsz, sent_num, -1, context_embd.shape[2])
                context_embd = torch.mean(context_embd, dim=2)
                context_mask = context_mask.reshape(bsz, sent_num, -1)
                device = context_mask.device
                mask_tensor = torch.tensor(context_mask, dtype=torch.float32, device=device)
                context_mask = torch.mean(mask_tensor, dim=2) > 0
                context_mask = torch.tensor(context_mask, dtype=torch.float32, device=device)
            elif self.context_doc_emded_type == "doc_rnn":
                context_embd = context_embd.reshape(bsz, sent_num, -1, context_embd.shape[2])
                context_mask = context_mask.reshape(bsz, sent_num, -1)

                context_embd = torch.mean(context_embd, dim=2)
                device = context_mask.device
                mask_tensor = torch.tensor(context_mask, dtype=torch.float32, device=device)
                context_mask = torch.mean(mask_tensor, dim=2) > 0
                device = context_mask.device
                context_mask = torch.tensor(context_mask, dtype=torch.float32, device=device)

                sentence_output = self.sentence_encoder(context_embd)

                context_embd = sentence_output[0]

                key_padding_mask = (1 - context_mask) > 0

            elif self.context_doc_emded_type == "weight_sum_of_selfattn":
                device = context_mask.device
                mask_tensor = torch.tensor(context_mask, dtype=torch.float32, device=device)
                context_mask = torch.mean(mask_tensor, dim=2) > 0
                context_embd = context_embd.reshape(bsz, sent_num, -1, context_embd.shape[2])
                context_embd = context_embd[:, :, 0, :]
                weight_matrix = self.ffn(torch.clone(context_embd))
                weight_matrix = torch.softmax(weight_matrix, dim=1)
                weight_matrix = weight_matrix.permute(0, 2, 1)
                context_embd = torch.bmm(weight_matrix, context_embd)

            if self.context_cross_attn:
                context_embd = self.cross_attn_forward(context_embd, query, self.cross_attn_layer)

        context_out, attn_weights = self.context_attn(
            query.permute(1, 0, 2),
            context_embd.permute(1, 0, 2),
            context_embd.permute(1, 0, 2),
            key_padding_mask,
        )
        context_out = self.proj_fc(context_out.permute(1, 0, 2))

        if self.context_global_dropout > 0:
            context_out = self.dropout_global_context(context_out)

        if self.context_joint_training:
            context_enabled = (context_mask.sum(dim=1, keepdim=True) > 0).to(torch.float)
            context_out = context_out * context_enabled.unsqueeze(1)
        return context_out, attn_weights
