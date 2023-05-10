"""
PyTorch BERT model for pipeline training.
"""


from collections import OrderedDict
from typing import Optional, Tuple

# pylint: disable=no-name-in-module
import torch
from torch import nn, Tensor, device
from torch.nn import CrossEntropyLoss

from core.extensions import PipelineModule, LayerSpec
from .bert_model import BertLayer, BertPooler, BertEmbeddings, BertPreTrainingHeads


class Embeddings(nn.Module):
    '''BERT Embeddings'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embeddings = BertEmbeddings(args)
        self.apply(self.init_weights)

    # pylint: disable=redefined-outer-name
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: device
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that
        future and masked tokens are ignored.
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :
            obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            if self.args.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                # in case past_key_values are used we need to add a
                # prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) "
                f"or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
        self,
        head_mask: Optional[Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :
                obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not
                (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x
            num_heads x seq_length x seq_length]` or list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        '''forward'''
        # pylint: disable=invalid-name
        input_ids = inputs[0]
        attention_mask = inputs[1]
        token_type_ids = inputs[2]
        position_ids = inputs[3]
        head_mask = inputs[4]
        inputs_embeds = inputs[5]

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.args.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        return (embedding_output, extended_attention_mask, head_mask)


class BertTransformerLayer(nn.Module):
    "Bert transformer layer"

    def __init__(self, args, idx):
        super().__init__()
        self.args = args
        self.layer = BertLayer(args)
        self.idx = idx
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        'forward'
        hidden_states = inputs[0]
        attention_mask = inputs[1]
        head_mask = inputs[2]
        layer_head_mask = head_mask[self.idx] if head_mask is not None else None

        layer_outputs = self.layer(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
        )
        return layer_outputs[0]


class BertPoolerLayer(nn.Module):
    "Bert pooler layer"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pooler = BertPooler(args)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        'forward'
        sequence_output = inputs[0]
        pooled_output = self.pooler(sequence_output)
        return (sequence_output, pooled_output)


class BertClsLayer(nn.Module):
    "Bert cls layer"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cls = BertPreTrainingHeads(args)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        'forward'
        sequence_output = inputs[0]
        pooled_output = inputs[1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        return (prediction_scores, seq_relationship_score)


class BertForPreTrainingPipe(PipelineModule):
    "pipeline bert model for pretrain"

    def __init__(self, args):
        'init'

        self.args = args
        layers = []
        layers.append(LayerSpec(Embeddings, args))
        for i in range(args.num_hidden_layers):
            layers.append(LayerSpec(BertTransformerLayer, args, i))
        layers.append(LayerSpec(BertPoolerLayer, args))
        layers.append(LayerSpec(BertClsLayer, args))

        super().__init__(layers=layers, loss_fn=self._loss_fn)

    def _loss_fn(self, inputs, target):
        '''loss func'''
        prediction_scores = inputs[0]
        seq_relationship_score = inputs[1]
        labels = target[0]
        next_sentence_label = target[1]

        total_loss = None
        masked_lm_loss = None
        next_sentence_loss = None
        loss_fct = CrossEntropyLoss()
        if labels is not None:
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.args.bert_vocab_size), labels.view(-1)
            )
        if next_sentence_label is not None:
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
        if masked_lm_loss and next_sentence_loss:
            total_loss = masked_lm_loss + next_sentence_loss
        else:
            total_loss = masked_lm_loss if masked_lm_loss else next_sentence_loss

        forward_out = OrderedDict()
        forward_out['loss'] = total_loss
        forward_out['backward_loss'] = total_loss
        # NOTE(zhengyijie): frame_size and tgt_size may not correct
        forward_out['frame_size'] = labels.float().sum()
        forward_out['tgt_size'] = labels.float().sum()
        forward_out['prediction_logits'] = prediction_scores
        forward_out['seq_relationship_logits'] = seq_relationship_score
        return forward_out
