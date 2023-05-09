import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2Model


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


class BaseTransformerDecoder(nn.Module):  # pragma: no cover
    def __init__(self) -> None:
        super().__init__()

    def cross_entropy(self, logits, labels):
        loss = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="mean")
        acc = labels == logits.argmax(2)
        return loss, acc.float().mean()

    def extract_channel_group_and_interleave(
        self, labels, start_channel, end_channel, apply_offset, codebook_size=1024
    ):
        """
        labels: batch_size x seq_len x channels
        """
        sub_labels = labels[..., start_channel:end_channel]
        return self.do_interleave(
            sub_labels, start_channel, end_channel, codebook_size, apply_offset
        )

    def do_interleave(
        self, seq, start_channel, end_channel, codebook_size, apply_offset
    ):
        if apply_offset:
            # Apply offsets for interleaving: code_id + codebook_size * channel_id
            offsets = (
                torch.arange(start_channel, end_channel, device=seq.device).view(
                    1, 1, -1
                )
                * codebook_size
            )
            # NOTE: there's a difference between seq += offset and seq = seq + offset
            # The former performs in-place add, which may modify the original buffer.
            # We should always use the latter to avoid unintended side effects.
            seq = seq + offsets.expand(seq.shape[0], seq.shape[1], -1)
        # batch_size x (seq_len * channels) interleaved
        seq = seq.reshape(seq.shape[0], -1)
        return seq

    def drop_last_element(self, seq):
        return seq[:, :-1]

    def prepend_start_token(self, seq, sos_id):
        start_token = torch.full((seq.size(0), 1), sos_id, device=seq.device).long()
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


#############################################
#             Text to Semantic           #
#############################################


class Text2SemanticTransformerDecoder(BaseTransformerDecoder):  # pragma: no cover
    def __init__(
        self,
        semantic_codebook_size: int = 1024,
        transformer_emb_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        transformer_ff_dim: int = 2048,
        semantic_seq_len: int = 250,
        use_text_emb_rvq: bool = True,
    ):
        super().__init__()
        self.semantic_codebook_size = semantic_codebook_size
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_ff_dim = transformer_ff_dim

        self.semantic_seq_len = semantic_seq_len
        self.semantic_sos_id = self.semantic_codebook_size
        self.semantic_emb_layer = nn.Embedding(
            self.semantic_codebook_size + 1, self.transformer_emb_dim
        )

        self.use_text_emb_rvq = use_text_emb_rvq

        n_positions = self.semantic_seq_len
        if self.use_text_emb_rvq:
            self.text_emb_layer = nn.Embedding(1024, self.transformer_emb_dim)
            n_positions += 6
        else:
            self.text_emb_layer = nn.Sequential(
                # MuLan embedding dim
                nn.Linear(128, self.transformer_emb_dim)
            )
            n_positions += 1

        self.gpt2_config = GPT2Config(
            vocab_size=self.semantic_codebook_size,
            n_positions=n_positions,
            n_embd=self.transformer_emb_dim,
            n_layer=self.transformer_layers,
            n_head=self.transformer_heads,
            n_inner=self.transformer_ff_dim,
        )
        self.decoder = GPT2Model(self.gpt2_config)
        self.fc_out = nn.Linear(self.transformer_emb_dim, self.semantic_codebook_size)

    def forward(self, semantic_tokens, text_input):
        transformer_input = self.prepend_start_token(
            semantic_tokens, self.semantic_sos_id
        )
        transformer_input = self.drop_last_element(transformer_input)
        transformer_input = self.semantic_emb_layer(transformer_input)
        text_emb = self.text_emb_layer(text_input)
        transformer_input = self.prepend_prefix(transformer_input, text_emb)
        folded_dec_out = self.decoder(inputs_embeds=transformer_input)[0]
        logits = self.fc_out(folded_dec_out)
        return logits

    def loss(self, semantic_tokens, text_input):
        logits = self.forward(semantic_tokens, text_input)
        logits = self.remove_prefix_from_output(logits, self.semantic_seq_len)
        loss, acc = self.cross_entropy(logits, semantic_tokens)
        return loss, acc

    def sample(self, text_input, temperature: float, sequence_length: int):
        bs = text_input.size(0)
        text_emb = self.text_emb_layer(text_input)
        semantic_sos = torch.full(
            (bs, 1), self.semantic_sos_id, device=text_input.device
        ).long()
        inputs_embeds = self.prepend_prefix(
            self.semantic_emb_layer(semantic_sos), text_emb
        )
        output_seq = torch.ones((bs, 0), device=text_input.device)

        past_key_values = None
        for _ in range(0, sequence_length):
            dec_output = self.decoder(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.fc_out(dec_output[0])
            past_key_values = dec_output[1]

            last_frame = logits[:, -1:, :]
            filtered_logits = top_k(last_frame.squeeze(1), thres=0.9)
            output_last_frame = gumbel_sample(filtered_logits, temperature).unsqueeze(
                dim=1
            )

            # output_last_frame =
            # Categorical(logits=(last_frame / temperature)).sample()
            output_seq = torch.cat([output_seq, output_last_frame], dim=1)
            # Prepare for next round
            inputs_embeds = self.semantic_emb_layer(output_last_frame)
        print(f"semantic output: {output_seq.size()}")
        return output_seq


#############################################
#         Text Semantic To Coarse        #
#############################################


class CoarseTransformerDecoder(BaseTransformerDecoder):  # pragma: no cover
    def __init__(
        self,
        acoustic_codebook_size: int = 1024,
        semantic_codebook_size: int = 1024,
        transformer_emb_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        transformer_ff_dim: int = 2048,
        acoustic_seq_len: int = 800,
        semantic_seq_len: int = 250,
        use_text_emb_rvq: bool = True,
        start_channel: int = 0,
        end_channel: int = 2,
    ):
        assert end_channel > start_channel
        super().__init__()
        self.acoustic_codebook_size = acoustic_codebook_size
        self.semantic_codebook_size = semantic_codebook_size
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.start_channel = start_channel
        self.end_channel = end_channel

        self.acoustic_seq_len = acoustic_seq_len * (
            self.end_channel - self.start_channel
        )
        self.semantic_seq_len = semantic_seq_len

        self.use_text_emb_rvq = use_text_emb_rvq

        self.acoustic_sos_id = self.acoustic_codebook_size * self.end_channel
        self.semantic_sos_id = self.semantic_codebook_size

        self.acoustic_emb_layer = nn.Embedding(
            self.acoustic_codebook_size * self.end_channel + 1, self.transformer_emb_dim
        )
        self.semantic_emb_layer = nn.Embedding(
            self.semantic_codebook_size + 1, self.transformer_emb_dim
        )
        n_positions = 1 + self.semantic_seq_len + self.acoustic_seq_len
        if self.use_text_emb_rvq:
            self.text_emb_layer = nn.Embedding(1024, self.transformer_emb_dim)
            n_positions += 6
        else:
            self.text_emb_layer = nn.Sequential(
                # MuLan embedding dim
                nn.Linear(128, self.transformer_emb_dim)
            )
            n_positions += 1

        self.gpt2_config = GPT2Config(
            vocab_size=self.acoustic_codebook_size,
            n_positions=n_positions,
            n_embd=self.transformer_emb_dim,
            n_layer=self.transformer_layers,
            n_head=self.transformer_heads,
            n_inner=self.transformer_ff_dim,
        )
        self.decoder = GPT2Model(self.gpt2_config)
        self.fc_out = nn.Linear(self.transformer_emb_dim, self.acoustic_codebook_size)

    def forward(self, acoustic_tokens, semantic_tokens, text_input):
        # acoustic tokens -> acoustic emb
        acoustic_tokens = self.extract_channel_group_and_interleave(
            acoustic_tokens.permute(0, 2, 1),
            self.start_channel,
            self.end_channel,
            codebook_size=self.acoustic_codebook_size,
            apply_offset=True,
        )
        acoustic_tokens = self.prepend_start_token(
            acoustic_tokens, self.acoustic_sos_id
        )
        acoustic_tokens = self.drop_last_element(acoustic_tokens)
        acounstic_emb = self.acoustic_emb_layer(acoustic_tokens)

        # semantic tokens -> semantic emb
        semantic_tokens = self.prepend_start_token(
            semantic_tokens, self.semantic_sos_id
        )
        semantic_emb = self.semantic_emb_layer(semantic_tokens)

        # text input -> text emb
        text_emb = self.text_emb_layer(text_input)

        transformer_input = self.prepend_prefix(acounstic_emb, semantic_emb)
        transformer_input = self.prepend_prefix(transformer_input, text_emb)

        folded_dec_out = self.decoder(inputs_embeds=transformer_input)[0]
        logits = self.fc_out(folded_dec_out)
        return logits

    def loss(self, acoustic_tokens, semantic_tokens, text_input):
        target = self.extract_channel_group_and_interleave(
            acoustic_tokens.permute(0, 2, 1),
            self.start_channel,
            self.end_channel,
            codebook_size=self.acoustic_codebook_size,
            apply_offset=False,
        )
        logits = self.forward(acoustic_tokens, semantic_tokens, text_input)
        logits = self.remove_prefix_from_output(logits, self.acoustic_seq_len)
        loss, acc = self.cross_entropy(logits, target)
        return loss, acc

    def sample(self, text_input, semantic_tokens, temperature, sequence_length):
        assert text_input.size(0) == semantic_tokens.size(0)
        bs = text_input.size(0)
        text_emb = self.text_emb_layer(text_input)
        semantic_tokens = self.prepend_start_token(
            semantic_tokens, self.semantic_sos_id
        )
        semantic_emb = self.semantic_emb_layer(semantic_tokens)
        coarse_sos = torch.full(
            (bs, 1), self.acoustic_sos_id, device=semantic_tokens.device
        ).long()
        coarse_emb = self.acoustic_emb_layer(coarse_sos)
        inputs_embeds = self.prepend_prefix(semantic_emb, text_emb)
        inputs_embeds = self.prepend_prefix(coarse_emb, inputs_embeds)
        output_seq = torch.ones((bs, 0), device=text_input.device)

        channel_offset = 0
        past_key_values = None
        for _ in range(0, sequence_length * (self.end_channel - self.start_channel)):
            dec_output = self.decoder(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.fc_out(dec_output[0])
            past_key_values = dec_output[1]

            last_frame = logits[:, -1:, :]
            filtered_logits = top_k(last_frame.squeeze(1), thres=0.9)
            output_last_frame = gumbel_sample(filtered_logits, temperature).unsqueeze(
                dim=1
            )

            # output_last_frame =
            # Categorical(logits=(logits[:, -1:, :] / temperature)).sample()
            output_seq = torch.cat([output_seq, output_last_frame], dim=1)
            # Prepare for next round
            output_offsetted = (
                output_last_frame
                + (self.start_channel + channel_offset) * self.acoustic_codebook_size
            )
            inputs_embeds = self.acoustic_emb_layer(output_offsetted)
            channel_offset = (channel_offset + 1) % (
                self.end_channel - self.start_channel
            )

        output_seq = output_seq.view(
            bs, -1, self.end_channel - self.start_channel
        ).permute(0, 2, 1)
        print(f"coarse output: {output_seq.size()}")
        return output_seq


#############################################
#              Coarse To Fine            #
#############################################


class FineTransformerDecoder(BaseTransformerDecoder):  # pragma: no cover
    def __init__(
        self,
        acoustic_codebook_size: int = 1024,
        transformer_emb_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        transformer_ff_dim: int = 2048,
        acoustic_seq_len: int = 240,
        start_channel: int = 2,
        end_channel: int = 6,
    ):
        super().__init__()
        assert end_channel > start_channel and start_channel > 0
        self.acoustic_codebook_size = acoustic_codebook_size
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.start_channel = start_channel
        self.end_channel = end_channel

        self.coarse_seq_len = acoustic_seq_len * self.start_channel
        self.fine_seq_len = acoustic_seq_len * (self.end_channel - self.start_channel)

        self.fine_sos_id = self.acoustic_codebook_size * self.end_channel

        self.acoustic_emb_layer = nn.Embedding(
            self.acoustic_codebook_size * self.end_channel + 1, self.transformer_emb_dim
        )
        n_positions = self.coarse_seq_len + self.fine_seq_len

        self.gpt2_config = GPT2Config(
            vocab_size=self.acoustic_codebook_size,
            n_positions=n_positions,
            n_embd=self.transformer_emb_dim,
            n_layer=self.transformer_layers,
            n_head=self.transformer_heads,
            n_inner=self.transformer_ff_dim,
        )
        self.decoder = GPT2Model(self.gpt2_config)
        self.fc_out = nn.Linear(self.transformer_emb_dim, self.acoustic_codebook_size)

    def forward(self, acoustic_tokens):
        # acoustic tokens -> acoustic emb
        fine_tokens = self.extract_channel_group_and_interleave(
            acoustic_tokens.permute(0, 2, 1),
            self.start_channel,
            self.end_channel,
            codebook_size=self.acoustic_codebook_size,
            apply_offset=True,
        )
        coarse_tokens = self.extract_channel_group_and_interleave(
            acoustic_tokens.permute(0, 2, 1),
            0,
            self.start_channel,
            codebook_size=self.acoustic_codebook_size,
            apply_offset=True,
        )
        fine_tokens = self.prepend_start_token(fine_tokens, self.fine_sos_id)
        fine_tokens = self.drop_last_element(fine_tokens)

        coarse_emb = self.acoustic_emb_layer(coarse_tokens)
        fine_emb = self.acoustic_emb_layer(fine_tokens)
        transformer_input = self.prepend_prefix(fine_emb, coarse_emb)

        folded_dec_out = self.decoder(inputs_embeds=transformer_input)[0]
        logits = self.fc_out(folded_dec_out)
        return logits

    def loss(self, acoustic_tokens):
        target = self.extract_channel_group_and_interleave(
            acoustic_tokens.permute(0, 2, 1),
            self.start_channel,
            self.end_channel,
            codebook_size=self.acoustic_codebook_size,
            apply_offset=False,
        )
        logits = self.forward(acoustic_tokens)
        logits = self.remove_prefix_from_output(logits, self.fine_seq_len)
        loss, acc = self.cross_entropy(logits, target)
        return loss, acc

    def sample(self, coarse_tokens, temperature, sequence_length, seed_sequence=None):
        assert self.start_channel == coarse_tokens.size(1)
        bs = coarse_tokens.size(0)
        output_seq = torch.ones((bs, 0), device=coarse_tokens.device)
        samples_to_generate = sequence_length * (self.end_channel - self.start_channel)
        channel_offset = 0

        coarse_tokens = self.extract_channel_group_and_interleave(
            labels=coarse_tokens.permute(0, 2, 1),
            start_channel=0,
            end_channel=self.start_channel,
            apply_offset=True,
            codebook_size=self.acoustic_codebook_size,
        )
        fine_sos = torch.full(
            (bs, 1), self.fine_sos_id, device=coarse_tokens.device
        ).long()
        inputs_embeds = self.prepend_prefix(fine_sos, coarse_tokens)

        if seed_sequence is not None:
            # Expect seed_sequence is already flattened
            # with offset added: batch_size x seq_len
            inputs_embeds = self.prepend_prefix(seed_sequence, inputs_embeds)
            samples_to_generate -= seed_sequence.size(1)
            # Seed sequence may affect starting channel index
            channel_offset = seed_sequence.size(1) % (
                self.end_channel - self.start_channel
            )

        inputs_embeds = self.acoustic_emb_layer(inputs_embeds)

        past_key_values = None
        for _ in range(0, samples_to_generate):
            dec_output = self.decoder(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.fc_out(dec_output[0])
            past_key_values = dec_output[1]

            last_frame = logits[:, -1:, :]
            filtered_logits = top_k(last_frame.squeeze(1), thres=0.9)
            output_last_frame = gumbel_sample(filtered_logits, temperature).unsqueeze(
                dim=1
            )

            # output_last_frame =
            # Categorical(logits=(logits[:, -1:, :] / temperature)).sample()
            output_seq = torch.cat([output_seq, output_last_frame], dim=1)
            # Prepare for next round
            output_offsetted = (
                output_last_frame
                + (self.start_channel + channel_offset) * self.acoustic_codebook_size
            )
            inputs_embeds = self.acoustic_emb_layer(output_offsetted)
            channel_offset = (channel_offset + 1) % (
                self.end_channel - self.start_channel
            )

        output_seq = output_seq.view(
            bs, -1, self.end_channel - self.start_channel
        ).permute(0, 2, 1)
        print(f"fine output: {output_seq.size()}")
        return output_seq


#############################################
#        [LEGACY] Coarse to Fine            #
#############################################


class PositionalEncoding(nn.Module):  # pragma: no cover
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


class DecoderT(nn.Module):  # pragma: no cover
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
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.nhead = nhead
        self.nlayers = nlayers
        self.d_feedforward = d_feedforward
        self.seq_len = max_seq_len
        self.use_gpt = use_gpt
        assert self.use_gpt, "Non-GPT2 models no longer supported!"

        self.gpt2_config = GPT2Config(
            vocab_size=self.input_dim,
            n_positions=self.seq_len,
            n_embd=self.emb_dim,
            n_layer=self.nlayers,
            n_head=self.nhead,
            n_inner=self.d_feedforward,
        )
        self.dec_layer = GPT2Model(self.gpt2_config)

        self.emb_layer = nn.Embedding(self.input_dim, self.emb_dim)
        self.pos_enc_layer = PositionalEncoding(
            self.emb_dim, max_len=max_seq_len
        )  # TODO: set correct max_len

        self.fc_out = nn.Linear(self.emb_dim, self.output_dim)

    def forward(self, x, cond):
        emb = self.emb_layer(x)
        src = self.pos_enc_layer(emb)
        if cond is not None:
            # add cond embedding at start of seq
            src = torch.cat([cond, src], dim=1)
        # batch_size x seq_length x model_size
        folded_dec_out = self.dec_layer(inputs_embeds=src)[0]
        logits = self.fc_out(folded_dec_out)
        return logits

    def sample(self, prefix, sequence_length, temperature, start_channel_offset=0):
        ret = prefix.clone()
        inputs_embeds = self.pos_enc_layer(self.emb_layer(prefix))

        start_pos = ret.size(1)
        past_key_values = None
        channel_offset = start_channel_offset
        for _ in range(0, sequence_length):
            outputs = self.dec_layer(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.fc_out(outputs[0])
            past_key_values = outputs[1]

            last_frame = logits[:, -1:, :]
            filtered_logits = top_k(last_frame.squeeze(1), thres=0.9)
            output_last_frame = gumbel_sample(filtered_logits, temperature).unsqueeze(
                dim=1
            )
            # last_frame_probs =
            # torch.nn.functional.softmax(last_frame / temperature, dim=-1)
            # output_onehot =  OneHotCategorical(probs=last_frame_probs).sample()
            # output = torch.argmax(output_onehot, dim=2)
            ret = torch.cat([ret, output_last_frame], dim=1)
            # Prepare for next round
            output_offsetted = (
                output_last_frame
                + (self.start_channel + channel_offset) * self.output_dim
            )
            inputs_embeds = self.pos_enc_layer(
                self.emb_layer(output_offsetted), start_pos=start_pos
            )
            start_pos += output_offsetted.size(1)
            channel_offset = (channel_offset + 1) % (
                self.end_channel - self.start_channel
            )
        return ret


class SoundStreamTransformerDecoder(nn.Module):  # pragma: no cover
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
        transformer_ff_dim: int = 1024,
        max_sequence_length: int = 64,
        codebook_size: int = 1024,
        start_channel: int = 0,
        end_channel: int = 2,
        use_gpt: bool = True,
        use_cond: bool = True,  # If False, ignore conditioning embeddings
        interleave: bool = True,
        cond_dropout: float = 0.0,
    ):
        super().__init__()
        self.cond_input_emb_dim = cond_input_emb_dim
        self.transformer_emb_dim = transformer_emb_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ff_dim = transformer_ff_dim
        self.codebook_size = codebook_size
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.use_gpt = use_gpt
        self.use_cond = use_cond
        self.interleave = interleave
        assert self.interleave, "Non-interleaving is no longer supported!"
        self.cond_dropout = cond_dropout

        # Need enough embeddings to model the channels being predicted,
        # plus the prefix channels used as conditioning,
        # plus the start token (the last embedding ID)
        self.transformer_num_embeddings = self.codebook_size * self.end_channel + 1
        self.sos_id = self.transformer_num_embeddings - 1

        self.cond_emb_layer = nn.Sequential(
            nn.Linear(self.cond_input_emb_dim, self.transformer_emb_dim),
            nn.Dropout(self.cond_dropout),
        )
        self.dec = DecoderT(
            emb_dim=self.transformer_emb_dim,
            input_dim=self.transformer_num_embeddings,
            nhead=self.transformer_heads,
            nlayers=self.transformer_layers,
            output_dim=self.codebook_size,
            d_feedforward=self.transformer_ff_dim,
            # max_seq_len is the prefix length plus the main length,
            # plus the conditioning vector length (assumed to be 1 for now)
            max_seq_len=self.end_channel * max_sequence_length + 1,
            use_gpt=self.use_gpt,
            start_channel=self.start_channel,
            end_channel=self.end_channel,
        )

    def forward(self, x, cond_embeddings):
        if not self.use_cond:
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
            # NOTE: there's a difference between seq += offset and seq = seq + offset
            # The former performs in-place add, which may modify the original buffer.
            # We should always use the latter to avoid unintended side effects.
            seq = seq + offsets.expand(seq.shape[0], seq.shape[1], -1)
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

    def loss(self, labels, cond_embeddings):
        if self.use_cond:
            cond_embeddings = self.cond_emb_layer(cond_embeddings)
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
        logits = self.dec(transformer_input, cond_embeddings)
        logits = self.remove_prefix_from_output(logits, seq_length)
        loss, acc = self.flattened_cross_entropy(logits, target_labels)
        return loss, acc

    def sample(
        self, coarse_tokens, temperature, sequence_length=240, seed_sequence=None
    ):
        bs = coarse_tokens.size(0)
        fine_sos = torch.full((bs, 1), self.sos_id, device=coarse_tokens.device).long()

        coarse_tokens = coarse_tokens.permute(0, 2, 1)
        prev_coarse_tokens = self.extract_channel_group_and_interleave(
            coarse_tokens, 0, self.start_channel, apply_offset=True
        )
        prefix = self.prepend_prefix(fine_sos, prev_coarse_tokens)

        num_channels = self.end_channel - self.start_channel
        seq_len = sequence_length * num_channels
        samples_to_generate = seq_len
        start_channel_offset = 0
        if seed_sequence is not None:
            # Expect seed_sequence is already flattened: batch_size x seq_len
            prefix = self.prepend_prefix(seed_sequence, prefix)
            samples_to_generate -= seed_sequence.size(1)
            # Seed sequence may affect starting channel index
            start_channel_offset = seed_sequence.size(1) % (
                self.end_channel - self.start_channel
            )
        output = self.dec.sample(
            prefix, samples_to_generate, temperature, start_channel_offset
        )
        # prefix may not be in target space, need to account for that
        output = self.remove_prefix_from_output(output, seq_len) % self.codebook_size
        # batch_size x (seq_len x num_channels) --> batch_size x num_channels x seq_len
        output = output.view(bs, -1, num_channels).permute(0, 2, 1)
        print(f"fine output: {output.shape}")
        return output
