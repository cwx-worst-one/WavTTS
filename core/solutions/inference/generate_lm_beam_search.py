''' beam search '''
# pylint: disable=line-too-long, cell-var-from-loop
import torch
from core.utils import FalconDict
from core.models.layers.transformer import CifSelfAttentionModule
from core.models.asr.cif_decoder import attention_bias_lower_triangle

INT_MAX = 2147483647
INF = 1.0 * 1e7


def log_prob_from_logits(logits, dim=2):
    '''log_prob_from_logits'''
    return logits - torch.logsumexp(logits, dim=dim, keepdim=True)


def gather_nd(x, indices):
    '''gather_nd'''
    expand_dims = False
    if len(x.size()) < 3:
        expand_dims = True
        x = x.unsqueeze(-1)
    newshape = indices.shape[:-1] + x.shape[indices.shape[-1] :]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    #pylint:disable=unnecessary-dunder-call
    out = torch.cat([x.__getitem__(tuple(i)) for i in indices])
    if expand_dims:
        return out.reshape(newshape).squeeze(-1)
    return out.reshape(newshape)


def reform(value, coordinates, shape, beam_size=None):
    '''reform'''
    flatten_value = torch.reshape(value, shape)
    if beam_size:
        reformed_value = torch.reshape(
            gather_nd(flatten_value, coordinates), (shape[0] * beam_size, shape[2], shape[3])
        )
    else:
        reformed_value = torch.reshape(
            gather_nd(flatten_value, coordinates), (shape[0] * shape[1], shape[2], shape[3])
        )
    return reformed_value


def compute_batch_indices(batch_size, beam_size, device='cuda'):
    """Computes the i'th coodinate that contains the batch index for gathers.

    Batch pos is a tensor like [[0,0,0,0,],[1,1,1,1],..]. It says which
    batch the beam item is in. This will create the i of the i,j coordinate
    needed for the gather.

    Args:
      batch_size: Batch size
      beam_size: Size of the beam.
    Returns:
      batch_pos: [batch_size, beam_size] tensor of ids
    """
    batch_pos = torch.arange(batch_size * beam_size, device=device) // beam_size
    batch_pos = torch.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos


def compute_topk_scores_and_seq(
    topk_seq,
    topk_scores,
    topk_log_probs,
    topk_finished,
    beam_size,
    batch_size,
    cache,
    i,
    hidden_size,
    cache_reform=False,
):
    """
    Compute topk scores and sequences
    """
    _, topk_ids = torch.topk(topk_scores, k=beam_size)
    batch_pos = compute_batch_indices(batch_size, beam_size, 'cuda')
    topk_coordinates = torch.stack([batch_pos, topk_ids], dim=2)

    topk_seq = gather_nd(topk_seq, topk_coordinates)
    topk_finished = gather_nd(topk_finished, topk_coordinates)
    topk_log_probs = gather_nd(topk_log_probs, topk_coordinates)

    if cache_reform:
        for key in cache:
            if "decoder_layer" in key:
                cache[key]["k"] = reform(
                    cache[key]["k"],
                    topk_coordinates,
                    (batch_size, 2 * beam_size, i + 1, hidden_size),
                    beam_size,
                )
                cache[key]["v"] = reform(
                    cache[key]["v"],
                    topk_coordinates,
                    (batch_size, 2 * beam_size, i + 1, hidden_size),
                    beam_size,
                )

    return topk_seq, topk_log_probs, topk_finished


class GenerateBeamSearch:
    '''base class of beam search of cif-based model'''

    def __init__(
        self,
        args,
        cfg,
        decoder_module,
        lm_solution=None,
        fst_solution=None,
        domain_fst_solution=None,
        fst_dict=None,
    ):
        """beam search required configuration"""
        self.args = args
        self.decoder_module = decoder_module
        self.lm_solution = lm_solution
        self.fst_solution = fst_solution
        self.domain_fst_solution = domain_fst_solution
        self.fst_dict = fst_dict

        self.beam_size = cfg.get('beam_size', 10)
        self.nbest = cfg.get('nbest', 1)
        self.output_temperature = cfg.get('output_temperature', 1.0)

        self.vocab_size = args.vocab_size
        self.eos_id = args.get('eos_id', 2)
        self.alpha = cfg.get('alpha', 1.0)

    def __call__(self, audio_embeds, audio_embeds_mask, text_embed_lookup, *args, **kwargs):
        '''the interface of beam search call'''
        batch_size = audio_embeds.size(0)
        decode_length = 2 * audio_embeds_mask.size(1) + self.args.extra_decode_length
        real_audio_length = torch.sum(audio_embeds_mask, dim=-1).unsqueeze(-1)

        # pad the input embeds and embeds_mask input to length of decode_length
        pad_embeds = (
            torch.zeros(
                [
                    audio_embeds.shape[0],
                    decode_length - audio_embeds.shape[1],
                    audio_embeds.shape[2],
                ]
            )
            .float()
            .cuda()
        )
        pad_embeds_mask = (
            torch.zeros([audio_embeds.shape[0], decode_length - audio_embeds.shape[1]])
            .float()
            .cuda()
        )
        audio_embeds = torch.cat([audio_embeds, pad_embeds], dim=1)
        audio_embeds_mask = torch.cat([audio_embeds_mask, pad_embeds_mask], dim=1)

        # expand to beamsize
        audio_embeds = self.expand_inputs_beamsize(audio_embeds)
        audio_embeds_mask = self.expand_inputs_beamsize(audio_embeds_mask, dim=2)

        running_hyp = self.get_init_hyp(batch_size, decode_length)
        init_alive_log_probs = torch.cat(
            [torch.zeros([batch_size, 1]), torch.full([batch_size, self.beam_size - 1], -99999.0)],
            dim=1,
        ).cuda()

        def beam_search_inner_loop(i, running_hyp):
            batch_size = running_hyp.batch_size
            alive_seq = running_hyp.alive_ids
            alive_log_probs = running_hyp.alive_log_probs
            cache = running_hyp.running_cache

            cur_ids = torch.reshape(alive_seq[:, :, -1], (batch_size * self.beam_size,))

            cur_logits, _, cache = self.decoder_module(
                i, cur_ids, audio_embeds, audio_embeds_mask, text_embed_lookup, cache
            )

            cur_logits = torch.reshape(cur_logits, (batch_size, self.beam_size, -1))
            candidate_log_probs = log_prob_from_logits(cur_logits * self.output_temperature)
            alive_log_probs = torch.where(
                i == real_audio_length, init_alive_log_probs, alive_log_probs
            )

            # Multiply the probabilites by the current probabilites of the beam.
            # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
            log_probs = candidate_log_probs + alive_log_probs.unsqueeze(-1)

            # add length penalty
            length_penalty = torch.pow((torch.tensor(5.0 + (i + 1)).cuda() / 6.0), self.alpha)
            log_probs = log_probs / length_penalty

            # Flatten output (beam_size, vocab_size) probs into a list of possibilities
            flat_cur_scores = torch.reshape(log_probs, [-1, self.beam_size * self.vocab_size])
            topk_scores, topk_ids = torch.topk(flat_cur_scores, k=2 * self.beam_size)

            # recover the length penalty
            topk_log_probs = topk_scores * length_penalty

            # Work out what beam the top probs are in.
            topk_beam_index = topk_ids // self.vocab_size
            topk_ids %= self.vocab_size  # Unflatten the ids

            # The next two steps are to create coordinates for gather_nd to pull
            # out the correct seqences from id's that we need to grow.
            batch_pos = compute_batch_indices(batch_size, 2 * self.beam_size, 'cuda')

            # top beams will give us the actual coordinates to do the gather.
            # stacking will create a tensor of dimension batch * beam , where the
            # last dimension contains the i,j gathering coordinates.
            topk_coordinates = torch.stack([batch_pos, topk_beam_index], dim=2)

            # Gather up the most probable beams both for the ids
            topk_seq = gather_nd(alive_seq, topk_coordinates)

            # Reform logits and bias logits and k,v in cache and bias cache
            for key in cache:
                if "decoder_layer" in key:
                    cache[key]["k"] = reform(
                        cache[key]["k"],
                        topk_coordinates,
                        (batch_size, self.beam_size, i + 1, self.args.hidden_size),
                        2 * self.beam_size,
                    )
                    cache[key]["v"] = reform(
                        cache[key]["v"],
                        topk_coordinates,
                        (batch_size, self.beam_size, i + 1, self.args.hidden_size),
                        2 * self.beam_size,
                    )

            # Append the most probable value
            topk_seq = torch.cat((topk_seq, topk_ids.unsqueeze(2).int()), dim=2)
            topk_finished = (topk_ids == self.eos_id).to(torch.float32)

            # grow alive
            for_alive_topk_scores = topk_scores + topk_finished * (-INF)
            alive_seq, alive_log_probs, _ = compute_topk_scores_and_seq(
                topk_seq,
                for_alive_topk_scores,
                topk_log_probs,
                topk_finished,
                self.beam_size,
                batch_size,
                cache,
                i,
                self.args.hidden_size,
                cache_reform=True,
            )

            # grow finished
            running_hyp['finished_seq'] = torch.cat(
                [
                    running_hyp['finished_seq'],
                    torch.zeros([batch_size, self.beam_size, 1]).to(torch.int32).cuda(),
                ],
                dim=2,
            )
            for_finished_topk_scores = topk_scores + (1.0 - topk_finished.to(torch.float32)) * (
                -INF
            )
            cur_finished_seq = torch.cat([running_hyp['finished_seq'], topk_seq], dim=1)
            cur_finished_scores = torch.cat(
                [running_hyp['finished_scores'], for_finished_topk_scores], dim=1
            )
            cur_finished_flags = torch.cat([running_hyp['finished_flags'], topk_finished], dim=1)
            finished_seq, finished_scores, finished_flags = compute_topk_scores_and_seq(
                cur_finished_seq,
                cur_finished_scores,
                cur_finished_scores,
                cur_finished_flags,
                self.beam_size,
                batch_size,
                cache,
                i,
                self.args.hidden_size,
            )

            running_hyp['alive_ids'] = alive_seq
            running_hyp['alive_log_probs'] = alive_log_probs
            running_hyp['running_cache'] = cache
            running_hyp['finished_seq'] = finished_seq
            running_hyp['finished_scores'] = finished_scores
            running_hyp['finished_flags'] = finished_flags

            return running_hyp

        for i in range(decode_length):
            running_hyp = beam_search_inner_loop(i, running_hyp)

        if self.nbest == 1:
            cond = torch.sum(running_hyp.finished_flags, dim=1, keepdim=True).unsqueeze(-1) > 0
            finished_seq = torch.where(cond, running_hyp.finished_seq, running_hyp.alive_ids)

            return finished_seq[:, 0, 1:]
        raise ValueError('Not implement')

    def expand_inputs_beamsize(self, inputs, dim=3):
        '''expand inputs'''
        if dim == 3:
            inputs = inputs.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
            outputs = torch.reshape(inputs, (-1, inputs.shape[-2], inputs.shape[-1]))
        elif dim == 2:
            inputs = inputs.unsqueeze(1).expand(-1, self.beam_size, -1)
            outputs = torch.reshape(inputs, (-1, inputs.shape[-1]))
        else:
            raise ValueError('Not support such dimension')

        return outputs

    def get_init_hyp(self, batch_size, decode_length):
        '''get initial hypothesis'''
        running_cache = {
            "decoder_layer_%d" % layer: {"k": None, "v": None}
            for layer in range(self.args.num_decoder_layers)
        }

        decode_length += self.args.get('extra_decode_length', None)

        decoder_self_attention_bias = attention_bias_lower_triangle(decode_length, device='cuda')
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
        running_cache['decoder_self_attention_bias'] = decoder_self_attention_bias

        finished_seq = torch.zeros(batch_size, self.beam_size, 1, dtype=torch.int32, device='cuda')
        finished_scores = (
            torch.ones(batch_size, self.beam_size, dtype=torch.float32, device='cuda') * -INF
        )
        finished_flags = torch.zeros(batch_size, self.beam_size, dtype=torch.int32, device='cuda')

        hyp = FalconDict(
            batch_size=batch_size,
            alive_ids=torch.ones(batch_size, self.beam_size, 1, dtype=torch.int32, device='cuda'),
            alive_log_probs=torch.cat(
                (
                    torch.zeros(batch_size, 1, dtype=torch.float32, device='cuda'),
                    torch.full(
                        (batch_size, self.beam_size - 1),
                        -99999.0,
                        dtype=torch.float32,
                        device='cuda',
                    ),
                ),
                axis=1,
            ),
            running_cache=running_cache,
            finished_seq=finished_seq,
            finished_scores=finished_scores,
            finished_flags=finished_flags,
        )
        return hyp
