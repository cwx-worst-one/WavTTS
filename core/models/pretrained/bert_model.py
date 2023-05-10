"""
PyTorch BERT model.
You can visit https://github.com/huggingface/transformers/blob/master/src/\
transformers/models/bert/modeling_bert.py to see the source code.
"""
# pylint:disable=too-many-lines


import math
from collections import OrderedDict
from typing import Optional, Tuple, List

# pylint: disable=no-name-in-module
import torch
from packaging import version
from torch import nn, Tensor, device
from torch.nn import CrossEntropyLoss

from core.models.layers.transformer import BiTransformerLayer
from core.models.layers.active_function import get_activation_fn
from .bert_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
    apply_chunking_to_forward,
)


# pylint: disable=invalid-name
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, args):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            args.bert_vocab_size, args.hidden_size, padding_idx=args.pad_token_id
        )
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.token_type_embeddings = nn.Embedding(args.type_vocab_size, args.hidden_size)

        # self.LayerNorm is not snake-cased to stick withTensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(args, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(args.max_position_embeddings).expand((1, -1))
        )
        # pylint: disable=unexpected-keyword-arg
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(
                    self.position_ids.size(), dtype=torch.long, device=self.position_ids.device
                ),
                persistent=False,
            )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        '''forward'''
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros,
        # which usually occurs when its auto-generated, registered buffer helps users when tracing
        # the model without passing token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    '''BertSelfAttention'''

    def __init__(self, args, position_embedding_type=None):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0 and not hasattr(args, "embedding_size"):
            raise ValueError(
                f"The hidden size ({args.hidden_size}) is not a multiple "
                f"of the number of attention "
                f"heads ({args.num_attention_heads})"
            )

        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            args, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = args.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * args.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = args.is_decoder

    def transpose_for_scores(self, x):
        '''transpose_for_scores'''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        '''forward'''
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor)
            # of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected
            # key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores + relative_position_scores_query + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is
            # (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    '''BertSelfOutput'''

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''forward'''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    '''BertAttention'''

    def __init__(self, args, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(args, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(args)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        '''prune_heads'''
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        '''forward'''
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    '''BertIntermediate'''

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(args.hidden_act)
        else:
            self.intermediate_act_fn = args.hidden_act

    def forward(self, hidden_states):
        '''forward'''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    '''BertOutput'''

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        '''forward'''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    '''BertLayer'''

    def __init__(self, args):
        super().__init__()
        self.chunk_size_feed_forward = args.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(args)
        self.is_decoder = args.is_decoder
        self.add_cross_attention = args.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = BertAttention(args, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(args)
        self.output = BertOutput(args)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        '''forward'''
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    f"cross-attention layers by setting `args.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        '''feed_forward_chunk'''
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    '''BertEncoder'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer_type = args.get("transformer_type", "BertLayer")
        if self.transformer_type != "BertLayer":
            # using BiTransformerLayer for encoder
            assert args.get("position_embedding_type", "absolute") == "absolute" and not args.get(
                "add_cross_attention", False
            )
        self.layer = nn.ModuleList(
            [self.encoder_layer(args) for _ in range(args.num_hidden_layers)]
        )
        self._register_load_state_dict_pre_hook(self._compatible_load_hook)

    def _compatible_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''
        compatible for load previous checkpoint.
        '''
        if self.transformer_type == "BertLayer":
            return
        for idx in range(self.args.num_hidden_layers):
            for wb in ["weight", "bias"]:
                # self attn qkv
                for name, new_name in zip(
                    ["query", "key", "value"], ['q_proj', 'k_proj', 'v_proj']
                ):
                    val = state_dict.pop(
                        prefix + "layer.{}.attention.self.{}.{}".format(idx, name, wb)
                    )
                    if val is not None:
                        state_dict[
                            prefix + "layer.{}.self_attn.{}.{}".format(idx, new_name, wb)
                        ] = val
                # self attn out proj
                val = state_dict.pop(prefix + "layer.{}.attention.output.dense.{}".format(idx, wb))
                if val is not None:
                    state_dict[prefix + "layer.{}.self_attn.out_proj_{}".format(idx, wb)] = val
                # self_attn_layer_norm
                val = state_dict.pop(
                    prefix + "layer.{}.attention.output.LayerNorm.{}".format(idx, wb)
                )
                if val is not None:
                    state_dict[prefix + "layer.{}.self_attn_layer_norm.{}".format(idx, wb)] = val
                # fc1
                val = state_dict.pop(prefix + "layer.{}.intermediate.dense.{}".format(idx, wb))
                if val is not None:
                    state_dict[prefix + "layer.{}.fc1.{}".format(idx, wb)] = val
                # fc2
                val = state_dict.pop(prefix + "layer.{}.output.dense.{}".format(idx, wb))
                if val is not None:
                    state_dict[prefix + "layer.{}.fc2.{}".format(idx, wb)] = val
                # final layernorm
                val = state_dict.pop(prefix + "layer.{}.output.LayerNorm.{}".format(idx, wb))
                if val is not None:
                    state_dict[prefix + "layer.{}.final_layer_norm.{}".format(idx, wb)] = val

    def encoder_layer(self, args):
        """endcoder layers"""
        if self.transformer_type != "BertLayer":
            return BiTransformerLayer(
                embed_dim=args.hidden_size,
                attention_heads=args.num_attention_heads,
                ffn_embed_dim=args.intermediate_size,
                attention_dropout=args.attention_probs_dropout_prob,
                hidden_dropout=args.hidden_dropout_prob,
                activation_dropout=0.0,
                activation=args.hidden_act,
                normalize_before=False,
                clamp_inf=False,
                squeeze_mem=False,
                layernorm_eps=args.layer_norm_eps,
            )
        return BertLayer(args)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        '''forward'''
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.args.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.transformer_type != "BertLayer":
                # this is a hack for get_extended_attention_mask
                encoder_padding_mask = attention_mask[:, 0, 0, :] / -10000.0
                layer_outputs = layer_module(
                    hidden_states,
                    encoder_padding_mask=encoder_padding_mask.bool(),
                    output_layer_result=True,
                    fused=self.args.fused_transformer,
                    batch_first=True,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.args.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        forward_out = OrderedDict()

        forward_out['last_hidden_state'] = hidden_states
        forward_out['past_key_values'] = next_decoder_cache
        forward_out['hidden_states'] = all_hidden_states
        forward_out['attentions'] = all_self_attentions
        forward_out['cross_attentions'] = all_cross_attentions

        return forward_out


class BertPooler(nn.Module):
    '''BertPooler'''

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        '''forward'''
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    '''BertPredictionHeadTransform'''

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        if isinstance(args.hidden_act, str):
            self.transform_act_fn = get_activation_fn(args.hidden_act)
        else:
            self.transform_act_fn = args.hidden_act
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)

    def forward(self, hidden_states):
        '''forward'''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    '''BertLMPredictionHead'''

    def __init__(self, args):
        super().__init__()
        self.transform = BertPredictionHeadTransform(args)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(args.hidden_size, args.bert_vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(args.bert_vocab_size))

        # Need a link between the two variables so that the bias is
        # correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        '''forward'''
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    '''BertOnlyMLMHead'''

    def __init__(self, args):
        super().__init__()
        self.predictions = BertLMPredictionHead(args)

    def forward(self, sequence_output):
        '''forward'''
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    '''BertOnlyNSPHead'''

    def __init__(self, args):
        super().__init__()
        self.seq_relationship = nn.Linear(args.hidden_size, 2)

    def forward(self, pooled_output):
        '''forward'''
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    '''BertPreTrainingHeads'''

    def __init__(self, args):
        super().__init__()
        self.predictions = BertLMPredictionHead(args)
        self.seq_relationship = nn.Linear(args.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        '''forward'''
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


def get_parameter_dtype(parameter):
    '''get_parameter_dtype'''

    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        # pylint: disable=protected-access
        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


class BertModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder,
    in which case a layer of cross-attention is added between the self-attention layers, following
    the architecture described in `Attention is all you need <https://arxiv.org/abs/1706.03762>`__
    by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
    Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument
    of the configuration set to :obj:`True`. To be used in a Seq2Seq model, the model needs to
    initialized with both :obj:`is_decoder` argument and :obj:`add_cross_attention` set to :
    obj:`True`; an :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, args, add_pooling_layer=True):
        super().__init__()
        self._register_load_state_dict_pre_hook(self.model_load_hook)
        self.args = args

        self.embeddings = BertEmbeddings(args)
        self.encoder = BertEncoder(args)

        self.pooler = BertPooler(args) if add_pooling_layer else None
        self.dtype = get_parameter_dtype(self)

        self.apply(self.init_weights)

    @staticmethod
    def model_load_hook(
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        """Transform huggingface-chkpt to dolphin-chkpt"""
        if not any(s.startswith('bert.') for s in list(state_dict.keys())):
            # maybe not a bert chkpt
            return

        lens = len(state_dict)
        for _ in range(lens):
            name = list(state_dict.keys())[0]
            val = state_dict.pop(name)
            new_name = name.replace('bert.', prefix)
            state_dict[new_name] = val

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

    def get_input_embeddings(self):
        '''get_input_embeddings'''
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        '''set_input_embeddings'''
        self.embeddings.word_embeddings = value

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

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.
        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids,
        # we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6
        # /mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))

        # fp16 compatibility
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either "
                f"`torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            # We can specify head_mask for each layer
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:
            `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:
            `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the
            encoder input. This mask is used in the cross-attention if the model is
            configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:
            `args.n_layers` with each tuple having 4 tensors of shape :obj:
            `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks.
            Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :
                obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :
                obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned
            and can be used to speed up decoding (see :obj:`past_key_values`).
        """
        # pylint:disable=too-many-branches
        # pylint:disable=too-many-locals
        output_attentions = (
            output_attentions if output_attentions is not None else self.args.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.args.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.args.use_return_dict

        if self.args.is_decoder:
            use_cache = use_cache if use_cache is not None else self.args.use_cache
        else:
            use_cache = False

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
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

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

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.args.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

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
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs

        forward_out = OrderedDict()
        forward_out['last_hidden_state'] = sequence_output
        forward_out['pooler_output'] = pooled_output
        forward_out['past_key_values'] = encoder_outputs['past_key_values']
        forward_out['hidden_states'] = encoder_outputs['hidden_states']
        forward_out['attentions'] = encoder_outputs['attentions']
        forward_out['cross_attentions'] = encoder_outputs['cross_attentions']
        return forward_out


class BertForPreTraining(nn.Module):
    '''BertForPreTraining'''

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.bert = BertModel(args)
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

    def get_output_embeddings(self):
        '''get_output_embeddings'''
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        '''set_output_embeddings'''
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``
            [-100, 0, ..., args.bert_vocab_size]`` (see ``input_ids`` docstring) Tokens with indices
            set to ``-100`` are ignored(masked), the loss is only computed for the tokens with
            labels in ``[0, ..., args.bert_vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss.
            Input should be a sequence pair(see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``:
                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        return_dict = return_dict if return_dict is not None else self.args.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

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

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs
            return ((total_loss,) + output) if total_loss is not None else output

        forward_out = OrderedDict()
        forward_out['loss'] = total_loss
        forward_out['backward_loss'] = total_loss
        # NOTE(zhengyijie): frame_size and tgt_size may not correct
        forward_out['frame_size'] = input_ids.float().sum()
        forward_out['tgt_size'] = labels.float().sum()
        forward_out['prediction_logits'] = prediction_scores
        forward_out['seq_relationship_logits'] = seq_relationship_score
        forward_out['hidden_states'] = outputs['hidden_states']
        forward_out['attentions'] = outputs['attentions']
        return forward_out
