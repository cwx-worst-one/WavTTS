"""Wav2letter decoders."""

import itertools as it
import torch


class W2lDecoder:
    """W2lDecoder"""

    def __init__(self, args, tgt_dict):
        '''init.'''
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        self.criterion_type = 1
        self.blank = (
            tgt_dict.index("<ctc_blank>") if "<ctc_blank>" in tgt_dict.indices else tgt_dict.bos()
        )

    # pylint: disable=no-self-use
    def decode(self, *_args):
        """decode"""
        return NotImplementedError

    def generate(self, models, sample, **_unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"}
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    @staticmethod
    def get_emissions(models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))


class W2lGreedyDecoder(W2lDecoder):
    """W2lGreedyDecoder"""

    def decode(self, emissions, lengths):
        """decode"""
        hypos = []
        for b in range(emissions.size(0)):
            lprobs = emissions[b, : lengths[b]]
            toks = lprobs.argmax(dim=-1).unique_consecutive()
            hypos.append([{'tokens': toks[toks != self.blank], 'score': 0}])
        return hypos
