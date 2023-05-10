'''
cif decoder
'''

import torch
from torch import nn
import torch.nn.functional as F
from core.models.layers.transformer import CifSelfAttentionModule


def attention_bias_lower_triangle(length, device=None):
    '''
    Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.

    Args:
        length: a Scalar.

    Returns:
        a `Tensor` with shape [1, 1, length, length].
    '''
    band = torch.tril(torch.ones(length, length), diagonal=0)
    if device is not None:
        band = band.to(device)
    band = -1e9 * (1.0 - band)
    return band.unsqueeze(0).unsqueeze(0)


class CifSelfAttentionDecoder(CifSelfAttentionModule):
    '''CifSelfAttentionDecoder'''

    def __init__(self, args):
        super().__init__(args, sign='decoder')

    def forward(self, inputs, bias, atten_cache):
        '''
        Args:
          decoder_input: a Tensor
          decoder_self_attention_bias: bias Tensor for self-attention
          save_weights_to: an optional dictionary to capture attention weights
            for vizualization; the weights tensor will be appended there under
            a string key created from the variable scope (including name).

        Returns:
          y: a Tensors
        '''
        dec_inp = inputs
        for layer in range(self.args.num_decoder_layers):
            if atten_cache is not None:
                dec_inp, layer_cache, _ = self.atten[layer](
                    dec_inp, bias, atten_cache['decoder_layer_%d' % layer], None
                )
                atten_cache['decoder_layer_%d' % layer] = layer_cache
            else:
                dec_inp, _, _ = self.atten[layer](dec_inp, bias, None, None)
            dec_inp = self.ffn[layer](dec_inp)
        return self.process(dec_inp), atten_cache


class CifDecoder(nn.Module):
    '''Transform the cif representations to transcriptions under the auto-regressive manner.'''

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_lookup = nn.parameter.Parameter(torch.empty(args.vocab_size, args.hidden_size))
        self.emb_proj = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.out_proj = nn.Linear(2 * args.hidden_size, args.vocab_size, bias=False)
        self.sa_decoder = CifSelfAttentionDecoder(args)
        self.reset_parameters()

    def reset_parameters(self):
        '''reset parameter'''
        nn.init.normal_(self.emb_lookup, 0.0, self.args.hidden_size**-0.5)

    def common_ar_decoder(self, the_input, var_emb, cif_outputs, decoder_self_attention_bias):
        '''common_ar_decoder'''
        decode_length = cif_outputs.shape[1]

        the_input = the_input.long()
        input_emb = var_emb[the_input.reshape(-1)].reshape(*the_input.shape, -1)
        if self.args.multiply_embedding_mode == "sqrt_depth":
            input_emb = input_emb * (self.args.hidden_size**0.5)

        # Get the right-shift cif output to match the info of the teacher forcing input
        shifted_cif_outputs = F.pad(cif_outputs, [0, 0, 1, 0])[:, :decode_length, :]

        # Concatencate the embedding and the output of cif part
        concatanated_tensor = torch.cat([shifted_cif_outputs, input_emb], dim=-1)
        decoder_inputs = self.emb_proj(concatanated_tensor)

        # San part
        if self.args.layer_prepostprocess_dropout != 0.0:
            decoder_inputs = F.dropout(
                decoder_inputs,
                self.args.layer_prepostprocess_dropout,
                training=self.training,
                inplace=True,
            )
        sa_outputs, _ = self.sa_decoder(decoder_inputs, decoder_self_attention_bias, None)

        # Projection part
        pre_softmax_inputs = torch.cat([sa_outputs, cif_outputs], dim=-1)
        the_logits = self.out_proj(pre_softmax_inputs)
        the_ids = torch.argmax(the_logits, dim=-1)

        return the_logits, the_ids

    def logits_decoder(self, prev_ids, prev_cif_outputs, cur_cif_outputs, bias, cache):
        '''Predicting current logits based on previous states without loop'''
        if prev_ids is None:
            batch_size = cur_cif_outputs.shape[0]
            prev_emb = self.emb_lookup[1, :].expand(batch_size, -1)
        else:
            prev_emb = self.emb_lookup[prev_ids.long(), :]
        if self.args.multiply_embedding_mode == "sqrt_depth":
            prev_emb = prev_emb * (self.args.hidden_size**0.5)

        # Concatenate the embedding and the output of encoder
        # Then project the concatenated tensor to the dim of hidden_size
        if prev_cif_outputs is None:
            concatanated_tensor = F.pad(prev_emb, (self.args.hidden_size, 0))
        else:
            concatanated_tensor = torch.cat([prev_cif_outputs, prev_emb], dim=1)
        decoder_inputs = self.emb_proj(concatanated_tensor)
        decoder_inputs = decoder_inputs.unsqueeze(1)

        # Conduct dropout
        if self.args.layer_prepostprocess_dropout != 0.0:
            decoder_inputs = F.dropout(
                decoder_inputs,
                self.args.layer_prepostprocess_dropout,
                training=self.training,
                inplace=True,
            )

        # self-attention part
        sa_outputs, cache = self.sa_decoder(decoder_inputs, bias, cache)
        sa_outputs = sa_outputs.squeeze(1)

        pre_softmax_inputs = torch.cat([sa_outputs, cur_cif_outputs], dim=1)
        cur_logits = self.out_proj(pre_softmax_inputs)
        return cur_logits, cache

    def logits_loop_decoder(
        self,
        i,
        prev_ids,
        cif_outputs,
        cache,
    ):
        '''
        Predicting current logits based on previous states

        Args:
          i: loop index
          prev_ids: The ids of previous step. [batch_size]
          cache: a dict used for storing the decoder state (k, v...), and so on...
          cif_outputs: The outputs of the cif part, with shape
                       [batch_size, cif_output_length, hidden_size]
          encoder_outputs: as the name
          concat_prev_logits: Whether concatenating the previous logits

        Returns:
          i, next_ids, next_states, logits.
        '''
        cur_logits, cache = self.logits_decoder(
            prev_ids,
            prev_cif_outputs=None if i == 0 else cif_outputs[:, i - 1, :],
            cur_cif_outputs=cif_outputs[:, i, :],
            bias=cache['decoder_self_attention_bias'][:, :, i : i + 1, : i + 1],
            cache=cache,
        )
        # Refresh the elements
        cur_ids = cur_logits.argmax(dim=-1)

        return cur_logits, cur_ids, cache

    def forward(self, cif_outputs, targets=None, is_training=False):
        '''
        Args:
          cif_outputs: with shape [batch_size, cif_output_length, hidden_size]
          targets: with shape [batch_size, target_length]

        Returns:
          logits: the softmax function input, with shape [batch_size, cif_output_length, vocab_size]
        '''
        var_emb = self.emb_lookup
        decode_length = cif_outputs.shape[1]
        # is_training = self.training

        # Get the attention bias of the decoder san
        decoder_self_attention_bias = attention_bias_lower_triangle(decode_length, var_emb.device)
        if self.args.get('limited_decoder_states', 0):
            limited_decoder_states = self.args.limited_decoder_states
            masking_matrix = torch.ones((1, 1, decode_length, decode_length))
            masking_matrix_left_all = 1.0 - torch.triu(masking_matrix, 1)
            masking_matrix_left_limited = 1.0 - torch.triu(masking_matrix, -limited_decoder_states)
            decoder_self_attention_bias = masking_matrix_left_all - masking_matrix_left_limited
            # Note, should convert to 0 and -Inf for softmax
            decoder_self_attention_bias = (1.0 - decoder_self_attention_bias) * -1e9
            decoder_self_attention_bias = decoder_self_attention_bias.cuda()

        if self.args.proximity_bias:
            decoder_self_attention_bias = (
                decoder_self_attention_bias
                + CifSelfAttentionModule.attention_bias_proximal(
                    decode_length, decoder_self_attention_bias.device
                )
            )

        if is_training and (self.args.use_teacher_forcing or self.args.use_pss):
            assert not (self.args.use_teacher_forcing and self.args.use_pss)

            # Get the teacher forcing input and corresponding embedding
            teacher_forcing_input = F.pad(targets, [1, 0], value=1)[:, :decode_length]

            # Parallel Schedule Sampling
            if self.args.use_pss:
                raise NotImplementedError('pss not support yet')
            logits, _ = self.common_ar_decoder(
                teacher_forcing_input, var_emb, cif_outputs, decoder_self_attention_bias
            )
        else:
            num_layers = self.args.num_decoder_layers or self.args.num_hidden_layers
            prev_ids = None
            cache = {
                "decoder_layer_%d" % layer: {"k": None, "v": None} for layer in range(num_layers)
            }
            cache['decoder_self_attention_bias'] = decoder_self_attention_bias

            all_logits = []
            for i in range(decode_length):
                prev_logits, prev_ids, cache = self.logits_loop_decoder(
                    i, prev_ids, cif_outputs, cache
                )
                all_logits.append(prev_logits.unsqueeze(1))

            logits = torch.cat(all_logits, dim=1)

        return logits


class GenerateDecoder(nn.Module):
    '''Transform audio embeds to transcriptions under the auto-regressive manner.'''

    def __init__(self, args, sign='decoder'):
        super().__init__()
        self.args = args
        self.sign = sign

        assert self.sign == 'gen_lm'
        vocab_size = args.vocab_size
        self.out_proj = nn.Linear(args.hidden_size, vocab_size, bias=False)
        self.sa_decoder = CifSelfAttentionDecoder(args)

    def logits_decoder(self, input_embedding, bias, cache):
        '''Predicting current logits based on previous states without loop'''
        decoder_inputs = input_embedding

        # Conduct dropout
        if self.args.layer_prepostprocess_dropout != 0.0:
            decoder_inputs = F.dropout(
                decoder_inputs,
                self.args.layer_prepostprocess_dropout,
                training=self.training,
                inplace=True,
            )

        # self-attention part
        sa_outputs, cache = self.sa_decoder(decoder_inputs, bias, cache)

        sa_outputs = sa_outputs.squeeze(1)
        cur_logits = self.out_proj(sa_outputs)
        return cur_logits, cache

    def logits_loop_decoder(
        self,
        i,
        prev_ids,
        embeds,
        embeds_mask,
        text_embed_lookup,
        cache,
    ):
        '''
        Predicting current logits based on previous states

        Args:
          i: loop index
          prev_ids: The ids of previous step. [batch_size]
          embeds: the given acoustic embeds and zero placeholder.
          embeds_mask: the mask of given acoustic embeds and zero placeholder.
          text_embed_lookup: the text embedding layer of inputs.
          cache: a dict used for storing the decoder state (k, v...), and so on...

        Returns:
          i, next_ids, next_states, logits.
        '''

        acoustic_length = embeds_mask.sum(-1)
        # give the start of sentence position (sos to represent the start of recognition)
        prev_ids = torch.where(i == acoustic_length, torch.ones_like(prev_ids), prev_ids)
        # give the invalid acoustic position
        prev_ids = torch.where(i >= acoustic_length, prev_ids, torch.zeros_like(prev_ids))
        # inputs embedding
        if len(acoustic_length.shape) == 1:
            acoustic_length = acoustic_length.unsqueeze(-1)
        input_embedding = torch.where(
            i >= acoustic_length,
            text_embed_lookup(prev_ids.reshape(-1)).reshape(*prev_ids.shape, -1),
            embeds[:, i],
        )

        cur_logits, cache = self.logits_decoder(
            input_embedding.unsqueeze(1),
            bias=cache['decoder_self_attention_bias'][:, :, i : i + 1, : i + 1],
            cache=cache,
        )
        # Refresh the elements
        cur_ids = cur_logits.argmax(dim=-1)

        return cur_logits, cur_ids, cache

    def get_decoder_self_attention_bias(self, decode_length, embeds):
        # Get the attention bias of the decoder san
        decoder_self_attention_bias = attention_bias_lower_triangle(decode_length, embeds.device)

        if self.args.proximity_bias:
            decoder_self_attention_bias = (
                decoder_self_attention_bias
                + CifSelfAttentionModule.attention_bias_proximal(
                    decode_length, decoder_self_attention_bias.device
                )
            )
        return decoder_self_attention_bias

    def forward(self, embeds, embeds_mask, is_training=False, text_embed_lookup=None):
        '''
        Args:
          embeds: concatanated audio and text embeds

        Returns:
          logits: the softmax function input, with shape [batch_size, decode_length, vocab_size]
        '''
        if is_training:
            # definite decode_length for training
            decode_length = embeds.shape[1]
            decoder_self_attention_bias = self.get_decoder_self_attention_bias(
                decode_length, embeds
            )

            # San part
            if self.args.layer_prepostprocess_dropout != 0.0:
                decoder_inputs = F.dropout(
                    embeds,
                    self.args.layer_prepostprocess_dropout,
                    training=self.training,
                    inplace=True,
                )
            sa_outputs, _ = self.sa_decoder(decoder_inputs, decoder_self_attention_bias, None)

            # Projection part
            logits = self.out_proj(sa_outputs)
        else:
            num_layers = self.args.num_decoder_layers
            prev_ids = torch.zeros([embeds.shape[0]]).int().cuda()
            cache = {
                "decoder_layer_%d" % layer: {"k": None, "v": None} for layer in range(num_layers)
            }

            decode_length = embeds.size()[1] * 2
            decode_length += self.args.get('extra_decode_length', 20)
            decoder_self_attention_bias = self.get_decoder_self_attention_bias(
                decode_length, embeds
            )
            cache['decoder_self_attention_bias'] = decoder_self_attention_bias

            # pad the input embeds and embeds_mask input to length of decode_length
            pad_embeds = (
                torch.zeros([embeds.shape[0], decode_length - embeds.shape[1], embeds.shape[2]])
                .float()
                .cuda()
            )
            pad_embeds_mask = (
                torch.zeros([embeds.shape[0], decode_length - embeds.shape[1]]).float().cuda()
            )
            embeds = torch.cat([embeds, pad_embeds], dim=1)
            embeds_mask = torch.cat([embeds_mask, pad_embeds_mask], dim=1)

            all_logits = []
            for i in range(decode_length):
                prev_logits, prev_ids, cache = self.logits_loop_decoder(
                    i, prev_ids, embeds, embeds_mask, text_embed_lookup, cache
                )
                all_logits.append(prev_logits.unsqueeze(1))

            logits = torch.cat(all_logits, dim=1)

        return logits
