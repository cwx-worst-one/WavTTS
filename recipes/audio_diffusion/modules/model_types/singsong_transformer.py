import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from tqdm import tqdm
from transformers import T5Config, T5Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2Model


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, max_len: int = 64):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model)
        )
        pe = torch.zeros(max_len, dim_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len * #vectors, model_size]
        """
        x = x + self.pe[start_pos : start_pos + x.size(1)]
        return x


# For the attention mechanism, it creates the mask: at any index,
# the attention can visit all the frames up the index, but nothing beyond it
def generate_square_subsequent_mask(seq_length, device):
    return torch.triu(
        torch.ones((seq_length, seq_length), device=device) * float("-inf"), diagonal=1
    )


class DecoderT(nn.Module):
    def __init__(
        self,
        emb_dim,
        input_dim,
        nhead,
        nlayers,
        output_dim,
        start_channel,
        end_channel,
        d_feedforward=1024,
        max_seq_len=64,
        use_gpt=True,
        use_t5=False,
        decoder_nlayers=None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.nhead = nhead
        self.nlayers = nlayers
        self.decoder_nlayers = decoder_nlayers
        self.d_feedforward = d_feedforward
        self.seq_len = max_seq_len
        self.use_gpt = use_gpt
        self.use_t5 = use_t5
        assert self.use_gpt or self.use_t5, "Only T5 and GPT2 models are supported!"
        assert not (
            self.use_gpt and self.use_t5
        ), "You can only use one of T5 or GPT2 models"

        if self.use_gpt:
            self.gpt2_config = GPT2Config(
                vocab_size=self.input_dim,
                n_positions=self.seq_len,
                n_embd=self.emb_dim,
                n_layer=self.nlayers,
                n_head=self.nhead,
                n_inner=self.d_feedforward,
            )
            self.dec_layer = GPT2Model(self.gpt2_config)
        else:
            # TODO: Figure out what input_dim should be given semantic tokens.
            # Config takes only single vocab_size, which means that
            # inputs and outputs need to be in the same language space
            self.t5_config = T5Config(
                vocab_size=self.input_dim,
                n_positions=self.seq_len,  # TODO: correct?
                d_model=self.emb_dim,
                d_kv=(int)(
                    self.emb_dim / self.nhead
                ),  # TODO: recheck, but it looks like we need specify it
                num_layers=self.nlayers,
                num_decoder_layers=self.decoder_nlayers,
                num_heads=self.nhead,
                d_ff=self.d_feedforward,
            )
            # TODO: investigate a difference between T5 and T5ForCondtionalGeneration
            self.dec_layer = T5Model(self.t5_config)

        self.emb_layer = nn.Embedding(self.input_dim, self.emb_dim)
        self.pos_enc_layer = PositionalEncoding(
            self.emb_dim, max_len=max_seq_len
        )  # TODO: set correct max_len

        self.fc_out = nn.Linear(self.emb_dim, self.output_dim)

    def forward(self, x, cond):
        if self.use_t5:
            assert cond is not None, "Conditioning is required if using T5"

        emb = self.emb_layer(x)
        src = self.pos_enc_layer(emb)
        if cond is not None:
            if self.use_gpt:
                # add cond embedding at start of seq
                src = torch.cat([cond, src], dim=1)
            else:
                cond_emb = self.emb_layer(cond)
                cond = self.pos_enc_layer(
                    cond_emb
                )  # TODO: is this what we should be doing?
        # batch_size x seq_length x model_size
        if self.use_gpt:
            folded_dec_out = self.dec_layer(inputs_embeds=src)[0]
        else:  # when using T5:
            folded_dec_out = self.dec_layer(
                inputs_embeds=cond, decoder_inputs_embeds=src
            )[0]
        logits = self.fc_out(folded_dec_out)
        return logits

    def sample(
        self,
        cond_embeddings,
        prefix,
        num_outputs,
        sequence_length,
        temperature,
        start_channel_offset=0,
        log_interval=64,
    ):
        if self.use_t5:
            assert cond_embeddings is not None, "Conditioning is required if using T5"

        ret = prefix
        inputs_embeds = self.pos_enc_layer(self.emb_layer(prefix))
        if cond_embeddings is not None:
            if self.use_gpt:
                cond_embeddings = cond_embeddings.view(num_outputs, 1, -1)
                # add cond embedding at start of seq
                inputs_embeds = torch.cat([cond_embeddings, inputs_embeds], dim=1)
            else:
                cond_embeddings = self.emb_layer(cond_embeddings)
                cond_embeddings = self.pos_enc_layer(
                    cond_embeddings
                )  # TODO: is this what we should be doing?
        start_pos = ret.shape[1]
        past_key_values = None
        max_probs = None
        channel_offset = start_channel_offset
        for step in range(0, sequence_length):
            if self.use_gpt:
                outputs = self.dec_layer(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = self.dec_layer(
                    inputs_embeds=cond_embeddings,
                    decoder_inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            logits = self.fc_out(outputs[0])
            past_key_values = outputs[1]
            last_frame_probs = torch.nn.functional.softmax(
                logits[:, -1:, :] / temperature, dim=-1
            )
            if max_probs is None:
                max_probs, _ = torch.max(last_frame_probs, dim=-1)
            else:
                tmp_max, _ = torch.max(last_frame_probs, dim=-1)
                max_probs = torch.cat([max_probs, tmp_max], dim=1)
            output_onehot = OneHotCategorical(probs=last_frame_probs).sample()
            output = torch.argmax(output_onehot, dim=2)
            ret = torch.cat([ret, output], dim=1)
            # Prepare for next round
            output_offsetted = (
                output + (self.start_channel + channel_offset) * self.output_dim
            )
            inputs_embeds = self.pos_enc_layer(
                self.emb_layer(output_offsetted), start_pos=start_pos
            )
            start_pos += output_offsetted.shape[1]
            channel_offset = (channel_offset + 1) % (
                self.end_channel - self.start_channel
            )
            # if (step + 1) % log_interval == 0:
            #    print(f"Last {log_interval} outputs: {ret[:, -log_interval:]}")
            #    print(f"Last {log_interval} max probs: {max_probs[:, -log_interval:]}")
        # Last print if necessary
        # if remainder != 0:
        #    print(f"Last {remainder} outputs: {ret[:, -remainder:]}")
        #    print(f"Last {log_interval} max probs: {max_probs[:, -remainder:]}")
        return ret


class SoundStreamTransformerDecoder(nn.Module):
    """
    Simplified decoder class specific for SoundStream.
    The model predicts codes for channels from start_channel to end_channel.
    Ground-truth codes from channels 0 to start_channel are used as prefix.
    """

    def __init__(
        self,
        cond_input_emb_dim: int = 128,
        transformer_emb_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_decoder_layers: int = None,
        transformer_ff_dim: int = 1024,
        max_sequence_length: int = 64,
        codebook_size: int = 1024,
        start_channel: int = 0,
        end_channel: int = 2,
        use_gpt: bool = True,
        use_t5: bool = False,
        # TODO: create for_singsong flag instead of use_gpt and use_t5 ones
        use_cond: bool = True,  # If False, ignore conditioning embeddings
        use_semantic_tokens: bool = False,
        semantic_tokens_length: int = None,
        interleave: bool = True,
    ):
        super().__init__()
        self.cond_input_emb_dim = cond_input_emb_dim
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_decoder_layers = transformer_decoder_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.codebook_size = codebook_size
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.use_gpt = use_gpt
        self.use_t5 = use_t5
        self.use_cond = use_cond
        self.use_semantic_tokens = use_semantic_tokens
        self.semantic_tokens_length = semantic_tokens_length
        self.interleave = interleave
        assert self.interleave, "Non-interleaving is no longer supported!"
        if self.use_t5:
            assert not self.use_cond, "We can't use text conditioning with T5 model"
        if self.use_semantic_tokens:
            assert self.use_t5, "Only support semantic tokens with T5"
            assert (
                self.semantic_tokens_length is not None
            ), "Need to specify semantic tokens length to use them"

        # Need enough embeddings to model the channels being predicted,
        # plus the prefix channels used as conditioning,
        # plus the start token (the last embedding ID)
        if not self.use_semantic_tokens:
            self.transformer_num_embeddings = self.codebook_size * self.end_channel + 1
        else:
            self.transformer_num_embeddings = (
                self.codebook_size * (self.end_channel + 1) + 1
            )
        self.sos_id = self.transformer_num_embeddings - 1

        # For T5, we don't need additional emb_layer for conditiong
        if not self.use_t5:
            self.cond_emb_layer = nn.Sequential(
                nn.Linear(self.cond_input_emb_dim, self.transformer_emb_dim)
            )
        else:
            self.cond_emb_layer = None
        self.dec = DecoderT(
            emb_dim=self.transformer_emb_dim,
            input_dim=self.transformer_num_embeddings,
            nhead=self.transformer_heads,
            nlayers=self.transformer_layers,
            decoder_nlayers=self.transformer_decoder_layers,
            output_dim=self.codebook_size,
            d_feedforward=self.transformer_ff_dim,
            # max_seq_len is the prefix length plus the main length,
            # plus the conditioning vector length (assumed to be 1 for now)
            # plus semantic tokens length if using semantic tokens
            max_seq_len=self.end_channel * max_sequence_length
            + 1
            + (self.semantic_tokens_length if self.use_semantic_tokens else 0),
            use_gpt=self.use_gpt,
            use_t5=self.use_t5,
            start_channel=self.start_channel,
            end_channel=self.end_channel,
        )

    def forward(self, x, cond_embeddings):
        if not self.use_cond:
            assert not self.use_t5, "If using T5, we need to have conditioning passed"
            cond_embeddings = None
        # TODO: Check dimensionliaty of data coming
        x = x.permute(0, 2, 1)  # swap seq_len with channel dimensions
        x = x.reshape(x.shape[0], -1)
        output = self.dec(x, cond_embeddings)
        return output.permute(0, 2, 1)  # swap seq_len with channel dimensions

    def flattened_cross_entropy(self, logits, labels):
        flattened_logits = logits.reshape(-1, self.codebook_size)
        flattened_labels = labels.reshape(-1)
        loss = F.cross_entropy(flattened_logits, flattened_labels, reduction="mean")
        acc = flattened_labels == flattened_logits.argmax(1)
        return loss, acc.float().mean()

    def extract_channel_group_and_interleave(
        self, labels, start_channel, end_channel, apply_offset
    ):
        """
        labels: batch_size x seq_len x channels
        """
        sub_labels = labels[..., start_channel:end_channel]
        return self.do_interleave(sub_labels, start_channel, end_channel, apply_offset)

    def do_interleave(self, seq, start_channel, end_channel, apply_offset):
        if apply_offset:
            # Apply offsets for interleaving: code_id + codebook_size * channel_id
            offsets = (
                torch.arange(start_channel, end_channel, device=seq.device).view(
                    1, 1, -1
                )
                * self.codebook_size
            )
            seq += offsets.expand(seq.shape[0], seq.shape[1], -1)
        # batch_size x (seq_len * channels) interleaved
        seq = seq.reshape(seq.shape[0], -1)
        return seq

    def drop_last_element(self, seq):
        return seq[:, :-1]

    def prepend_start_token(self, seq):
        start_token = torch.full(
            (seq.size(0), 1), self.sos_id, device=seq.device
        ).long()
        seq = torch.cat([start_token, seq], dim=1)
        return seq

    def prepend_prefix(self, seq, prefix):
        """
        seq: batch_size x seq_len_1
        prefix: batch_size x seq_len_2
        """
        seq = torch.cat([prefix, seq], dim=1)
        return seq

    def remove_prefix_from_output(self, seq, seq_length):
        return seq[:, -seq_length:, ...]

    def loss(self, labels, cond_embeddings, labels_semantic=None, cond_semantic=None):
        if self.use_cond:
            if not self.use_t5:  # For T5, don't apply any emb_layer
                cond_embeddings = self.cond_emb_layer(cond_embeddings)
        else:
            if self.use_t5:
                assert (
                    cond_embeddings is not None
                ), "If using T5, we need to have conditioning passed"
                if self.use_semantic_tokens:
                    assert (labels_semantic is not None) and (
                        cond_semantic is not None
                    ), "Semantic tokens are required for labels and cond_embeddings"
            else:
                cond_embeddings = None
        # batch_size x channels x seq_len --> batch_size x seq_len x channels
        labels = labels.permute(0, 2, 1)
        loss = torch.zeros([1], device=labels.device)
        acc = torch.zeros([1], device=labels.device)

        target_labels = self.extract_channel_group_and_interleave(
            labels, self.start_channel, self.end_channel, apply_offset=False
        )
        seq_length = target_labels.shape[1]
        transformer_input = self.extract_channel_group_and_interleave(
            labels, self.start_channel, self.end_channel, apply_offset=True
        )
        transformer_input = self.prepend_start_token(transformer_input)
        transformer_input = self.drop_last_element(transformer_input)
        if self.start_channel > 0:
            prefix = self.extract_channel_group_and_interleave(
                labels, 0, self.start_channel, apply_offset=True
            )
            transformer_input = self.prepend_prefix(transformer_input, prefix)
        # batch_size x seq_len x classes
        if self.use_t5:
            cond_embeddings = cond_embeddings.permute(
                0, 2, 1
            )  # batch_size x seq_len x channels
            cond_embeddings = self.extract_channel_group_and_interleave(
                cond_embeddings, self.start_channel, self.end_channel, apply_offset=True
            )  # batch_size x seq_len * (end_channel-start)
            if self.use_semantic_tokens:
                semantic_offset_by = (
                    self.end_channel - self.start_channel
                ) * self.codebook_size
                cond_embeddings = torch.cat(
                    [cond_semantic + semantic_offset_by, cond_embeddings], dim=1
                )

        logits = self.dec(transformer_input, cond_embeddings)
        logits = self.remove_prefix_from_output(logits, seq_length)
        loss, acc = self.flattened_cross_entropy(logits, target_labels)
        return loss, acc

    def sample(
        self,
        cond_embeddings,
        labels,
        temperature,
        num_outputs=10,
        sequence_length=2048,
        device="cpu",
        seed_sequence=None,
        cond_semantic=None
        # Note: we don't need semantic tokens for labels
    ):
        if self.use_cond:
            if not self.use_t5:  # For T5, don't apply any emb_layer
                cond_embeddings = self.cond_emb_layer(cond_embeddings)
                cond_embeddings = cond_embeddings.repeat(num_outputs, 1, 1)
        else:
            if self.use_t5:
                assert (
                    cond_embeddings is not None
                ), "If using T5, we need to have conditioning passed"
                if self.use_semantic_tokens:
                    assert (
                        cond_semantic is not None
                    ), "Semantic tokens are required for cond_embeddings"
            else:
                cond_embeddings = None
        prefix = torch.full((num_outputs, 1), self.sos_id, device=device).long()
        if self.start_channel > 0:
            assert labels.shape[0] == 1, "Only support valid batch size 1 for now"
            # batch_size x channels x seq_len --> batch_size x seq_len x channels
            labels = labels.repeat(num_outputs, 1, 1)
            labels = labels.permute(0, 2, 1)
            prev_labels = self.extract_channel_group_and_interleave(
                labels, 0, self.start_channel, apply_offset=True
            )
            prefix = self.prepend_prefix(prefix, prev_labels)
        num_channels = self.end_channel - self.start_channel
        seq_len = sequence_length * num_channels
        samples_to_generate = seq_len
        start_channel_offset = 0
        if seed_sequence is not None:
            # Expect seed_sequence is already flattened: batch_size x seq_len
            prefix = self.prepend_prefix(seed_sequence, prefix)
            samples_to_generate -= seed_sequence.shape[1]
            # Seed sequence may affect starting channel index
            start_channel_offset = seed_sequence.shape[1] % (
                self.end_channel - self.start_channel
            )
        if self.use_t5:
            cond_embeddings = cond_embeddings.permute(0, 2, 1)
            cond_embeddings = self.extract_channel_group_and_interleave(
                cond_embeddings, self.start_channel, self.end_channel, apply_offset=True
            )
            if self.use_semantic_tokens:
                semantic_offset_by = (
                    self.end_channel - self.start_channel
                ) * self.codebook_size
                cond_embeddings = torch.cat(
                    [cond_semantic + semantic_offset_by, cond_embeddings], dim=1
                )
        output = self.dec.sample(
            cond_embeddings,
            prefix,
            num_outputs,
            samples_to_generate,
            temperature,
            start_channel_offset,
        )
        # prefix may not be in target space, need to account for that
        output = self.remove_prefix_from_output(output, seq_len) % self.codebook_size
        # batch_size x (seq_len x num_channels) --> batch_size x num_channels x seq_len
        output = output.view(num_outputs, -1, num_channels).permute(0, 2, 1)
        print(f"output: {output.shape}")
        return output


def temperature_sample(preds, temperature):
    last_frame_probs = torch.nn.functional.softmax(preds / temperature, dim=-1)
    output_onehot = OneHotCategorical(probs=last_frame_probs).sample()
    return torch.argmax(output_onehot, dim=2)


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class SingSongDecoderT(nn.Module):
    def __init__(
        self,
        emb_dim,
        input_dim,
        nhead,
        nlayers,
        output_dim,
        start_channel,
        end_channel,
        d_feedforward=1024,
        max_seq_len=64,
        use_gpt=True,
        use_t5=False,
        decoder_nlayers=None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.nhead = nhead
        self.nlayers = nlayers
        self.decoder_nlayers = decoder_nlayers
        self.d_feedforward = d_feedforward
        self.seq_len = max_seq_len
        self.use_gpt = use_gpt
        self.use_t5 = use_t5
        assert self.use_gpt or self.use_t5, "Only T5 and GPT2 models are supported!"
        assert not (
            self.use_gpt and self.use_t5
        ), "You can only use one of T5 or GPT2 models"

        if self.use_gpt:
            self.gpt2_config = GPT2Config(
                vocab_size=self.input_dim,
                n_positions=self.seq_len,
                n_embd=self.emb_dim,
                n_layer=self.nlayers,
                n_head=self.nhead,
                n_inner=self.d_feedforward,
            )
            self.dec_layer = GPT2Model(self.gpt2_config)
        else:
            # TODO: Figure out what input_dim should be given semantic tokens.
            # Config takes only single vocab_size, which means that
            # inputs and outputs need to be in the same language space
            self.t5_config = T5Config(
                vocab_size=self.input_dim,
                n_positions=self.seq_len,  # TODO: correct?
                d_model=self.emb_dim,
                d_kv=(int)(
                    self.emb_dim / self.nhead
                ),  # TODO: recheck, but it looks like we need specify it
                num_layers=self.nlayers,
                num_decoder_layers=self.decoder_nlayers,
                num_heads=self.nhead,
                d_ff=self.d_feedforward,
                feed_forward_proj="gated-gelu",
            )
            # TODO: investigate a difference between T5 and T5ForCondtionalGeneration
            self.dec_layer = T5Model(self.t5_config)

        self.emb_layer = nn.Embedding(self.input_dim, self.emb_dim)
        self.pos_enc_layer = PositionalEncoding(
            self.emb_dim, max_len=max_seq_len
        )  # TODO: set correct max_len

        self.fc_out = nn.Linear(self.emb_dim, self.output_dim)

    def forward(self, decoder_input, encoder_input):
        assert encoder_input is not None, "Conditioning is required if using T5"

        decoder_emb = self.emb_layer(decoder_input)
        encoder_emb = self.emb_layer(encoder_input)

        # batch_size x seq_length x model_size
        folded_dec_out = self.dec_layer(
            inputs_embeds=encoder_emb, decoder_inputs_embeds=decoder_emb
        )[0]
        logits = self.fc_out(folded_dec_out)
        return logits

    def sample(
        self,
        encoder_inputs,
        prefix,
        num_outputs,
        sequence_length,
        temperature,
        filter_thres=0.9,
    ):
        assert encoder_inputs is not None, "Conditioning is required if using T5"
        assert (
            encoder_inputs.min() >= self.output_dim * 2
        ), " semantic token id's must be offset by the coarse quantizers"

        result_token_ids = prefix
        semantic_seq_len = encoder_inputs.shape[1]
        decoder_inputs_emb = self.emb_layer(prefix)
        encoder_inputs_emb = self.emb_layer(encoder_inputs)

        start_pos = result_token_ids.shape[1]
        past_key_values = None
        channel_offset = 0
        for step in tqdm(range(0, sequence_length), desc="sampling s->sa"):
            outputs = self.dec_layer(
                inputs_embeds=encoder_inputs_emb,
                decoder_inputs_embeds=decoder_inputs_emb,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.fc_out(outputs[0])
            past_key_values = outputs[1]

            # temperature sampling
            last_frame = logits[:, -1:, :]
            # output = temperature_sample(last_frame, temperature)

            last_frame = last_frame.squeeze(dim=1)
            filtered_logits = top_k(last_frame, thres=filter_thres)
            output = gumbel_sample(filtered_logits, temperature)

            output = output.unsqueeze(dim=1)

            result_token_ids = torch.cat([result_token_ids, output], dim=1)

            # Prepare for next round
            if step < semantic_seq_len:
                output_offsetted = output + self.end_channel * self.output_dim
                assert (
                    output_offsetted.min() >= 1024 * 2
                ), "semantic token id's must be offset by 2 coarse quantizers"
            else:
                output_offsetted = (
                    output + (self.start_channel + channel_offset) * self.output_dim
                )
                channel_offset = (channel_offset + 1) % (
                    self.end_channel - self.start_channel
                )

            # TODO: T5 uses positional encoding internally!
            # How do we solve positioning with past_key_values/cache?
            decoder_inputs_emb = self.emb_layer(output_offsetted)
            start_pos += output_offsetted.shape[1]

        # crop only the coarse token id's
        return result_token_ids


class SingSong(nn.Module):
    """
    Simplified decoder class specific for SoundStream.
    The model predicts codes for channels from start_channel to end_channel.
    Ground-truth codes from channels 0 to start_channel are used as prefix.
    """

    def __init__(
        self,
        cond_input_emb_dim: int = 128,
        transformer_emb_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        transformer_decoder_layers: int = None,
        transformer_ff_dim: int = 1024,
        max_sequence_length: int = 64,
        codebook_size: int = 1024,
        start_channel: int = 0,
        end_channel: int = 2,
        use_gpt: bool = True,
        use_t5: bool = False,
        # TODO: create for_singsong flag instead of use_gpt and use_t5 ones
        use_cond: bool = True,  # If False, ignore conditioning embeddings
        use_semantic_tokens: bool = False,
        use_target_semantic: bool = False,
        semantic_tokens_length: int = None,
        interleave: bool = True,
    ):
        super().__init__()
        self.cond_input_emb_dim = cond_input_emb_dim
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_decoder_layers = transformer_decoder_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.codebook_size = codebook_size
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.n_channels = self.end_channel - self.start_channel
        self.use_gpt = use_gpt
        self.use_t5 = use_t5
        self.use_cond = use_cond
        self.use_semantic_tokens = use_semantic_tokens
        self.use_target_semantic = use_target_semantic
        self.semantic_tokens_length = semantic_tokens_length
        self.interleave = interleave
        assert self.interleave, "Non-interleaving is no longer supported!"

        # Need enough embeddings to model the channels being predicted,
        # plus the prefix channels used as conditioning,
        # plus the start token (the last embedding ID)
        # Assume that vocabulary size of semantic tokens is equal to self.codebook_size
        # we only keep Q=2
        self.transformer_num_embeddings = self.codebook_size * (self.n_channels + 1) + 1

        # TODO: logic for semantic token
        # semantic token vocab size = soundstream codebook size
        self.sos_id = self.transformer_num_embeddings - 1

        self.dec = SingSongDecoderT(
            emb_dim=self.transformer_emb_dim,
            input_dim=self.transformer_num_embeddings,
            nhead=self.transformer_heads,
            nlayers=self.transformer_layers,
            decoder_nlayers=self.transformer_decoder_layers,
            output_dim=self.codebook_size,
            d_feedforward=self.transformer_ff_dim,
            # max_seq_len is the prefix length plus the main length,
            # plus the conditioning vector length (assumed to be 1 for now)
            # plus semantic tokens length if using semantic tokens
            max_seq_len=self.semantic_tokens_length
            + 1
            + self.n_channels * max_sequence_length,
            use_gpt=self.use_gpt,
            use_t5=self.use_t5,
            start_channel=self.start_channel,
            end_channel=self.end_channel,
        )

    def forward(self, x, cond_embeddings):
        if not self.use_cond:
            assert not self.use_t5, "If using T5, we need to have conditioning passed"
            cond_embeddings = None
        # TODO: Check dimensionliaty of data coming
        x = x.permute(0, 2, 1)  # swap seq_len with channel dimensions
        x = x.reshape(x.shape[0], -1)
        output = self.dec(x, cond_embeddings)
        return output.permute(0, 2, 1)  # swap seq_len with channel dimensions

    def flattened_cross_entropy(self, logits, labels):
        flattened_logits = logits.reshape(-1, self.codebook_size)
        flattened_labels = labels.reshape(-1)
        loss = F.cross_entropy(flattened_logits, flattened_labels, reduction="mean")
        acc = flattened_labels == flattened_logits.argmax(1)
        return loss, acc.float().mean()

    def extract_channel_group_and_interleave(
        self, labels, start_channel, end_channel, apply_offset
    ):
        """
        labels: batch_size x seq_len x channels
        """
        sub_labels = labels[..., start_channel:end_channel]
        return self.do_interleave(sub_labels, start_channel, end_channel, apply_offset)

    def do_interleave(self, seq, start_channel, end_channel, apply_offset):
        if apply_offset:
            # Apply offsets for interleaving: code_id + codebook_size * channel_id
            offsets = (
                torch.arange(start_channel, end_channel, device=seq.device).view(
                    1, 1, -1
                )
                * self.codebook_size
            )
            seq += offsets.expand(seq.shape[0], seq.shape[1], -1)
        # batch_size x (seq_len * channels) interleaved
        seq = seq.reshape(seq.shape[0], -1)
        return seq

    def drop_last_element(self, seq):
        return seq[:, :-1]

    def prepend_start_token(self, seq):
        start_token = torch.full(
            (seq.size(0), 1), self.sos_id, device=seq.device
        ).long()
        seq = torch.cat([start_token, seq], dim=1)
        return seq

    def prepend_prefix(self, seq, prefix):
        """
        seq: batch_size x seq_len_1
        prefix: batch_size x seq_len_2
        """
        seq = torch.cat([prefix, seq], dim=1)
        return seq

    def remove_prefix_from_output(self, seq, seq_length):
        return seq[:, -seq_length:, ...]

    def loss(
        self,
        accompaniment_acoustic_token_ids,
        encoder_inputs,
        accompaniment_semantic_token_ids=None,
        vocal_semantic_token_ids=None,
    ):
        assert (
            encoder_inputs is not None
        ), "If using T5, we need to have conditioning passed"
        if self.use_semantic_tokens:
            assert (accompaniment_semantic_token_ids is not None) and (
                vocal_semantic_token_ids is not None
            ), "Semantic tokens are required for labels and cond_embeddings"
        # batch_size x channels x seq_len --> batch_size x seq_len x channels
        accompaniment_acoustic_token_ids = accompaniment_acoustic_token_ids.permute(
            0, 2, 1
        )
        loss = torch.zeros([1], device=accompaniment_acoustic_token_ids.device)
        acc = torch.zeros([1], device=accompaniment_acoustic_token_ids.device)

        # 1. make the accompaniment acoustic tokens unique
        # per quantizer ("interleaving")
        target_acc_acousic_token_ids = self.extract_channel_group_and_interleave(
            accompaniment_acoustic_token_ids,
            self.start_channel,
            self.end_channel,
            apply_offset=False,
        )

        # 2. prepare the target tensor, so that it models the (SA) part of (S-SA),
        # namely prepend the semantic token
        # id's to the accompaniment acoustic token id's
        target_token_ids = self.prepend_prefix(
            target_acc_acousic_token_ids, accompaniment_semantic_token_ids
        )

        # 2.5 apply interleaving to the accompaniment acoustic token id's
        interleaved_accompaniment_acoustic_token_ids = (
            self.extract_channel_group_and_interleave(
                accompaniment_acoustic_token_ids,
                self.start_channel,
                self.end_channel,
                apply_offset=True,
            )
        )

        # 3. add offsets to the semantics, so they don't clash with acoustic ones
        semantic_offset_by = self.n_channels * self.codebook_size
        # 4. prepend the semantic tokens with the acoustic tokens, and add the offset,
        # so that we indeed model (SA) in the (S-SA) problem
        decoder_input = torch.cat(
            [
                accompaniment_semantic_token_ids + semantic_offset_by,
                interleaved_accompaniment_acoustic_token_ids,
            ],
            dim=1,
        )

        # 5. prepend the SOS token to our concatenated semantic
        # and audio accompaniment token id's
        decoder_input = self.prepend_start_token(decoder_input)
        # 6. we have to drop the last one, to ensure equal sequence lengths
        decoder_input = self.drop_last_element(decoder_input)
        # assert decoder_input[0, 0] == self.sos_id

        # 7a. NOTE: We could re-use this later to model (SA-SA)
        # vocal_acoustic_token_ids = vocal_acoustic_token_ids.permute(0, 2, 1)
        # vocal_acoustic_token_ids = self.extract_channel_group_and_interleave(
        #     vocal_acoustic_token_ids, self.start_channel, self.end_channel,
        #   apply_offset=True
        # ) # batch_size x seq_len * (end_channel-start)

        # 7b. we feed the semantic token id's of the vocal audio to the T5 encoder.
        # because we use the same embedding layer, we offset these also like
        # the accompaniment semantic token id's
        encoder_inputs = vocal_semantic_token_ids + semantic_offset_by

        logits = self.dec(decoder_input, encoder_inputs)
        loss, acc = self.flattened_cross_entropy(logits, target_token_ids)
        return loss, acc, logits

    def sample(
        self,
        cond_embeddings,
        labels,
        temperature,
        num_outputs=10,
        sequence_length=2048,
        device="cpu",
    ):
        prefix = torch.full((num_outputs, 1), self.sos_id, device=device).long()
        semantic_token_ids = cond_embeddings

        seq_len = sequence_length * self.n_channels
        semantic_seq_len = semantic_token_ids.shape[1]
        samples_to_generate = semantic_seq_len + seq_len

        semantic_offset_by = self.n_channels * self.codebook_size
        encoder_inputs = semantic_token_ids + semantic_offset_by

        output = self.dec.sample(
            encoder_inputs, prefix, num_outputs, samples_to_generate, temperature
        )
        # prefix may not be in target space, need to account for that
        # this also truncates the semantic token id's
        output = self.remove_prefix_from_output(output, seq_len) % self.codebook_size
        # batch_size x (seq_len x num_channels) --> batch_size x num_channels x seq_len
        output = output.view(num_outputs, -1, self.n_channels).permute(0, 2, 1)
        print(f"output: {output.shape}")
        return output
