import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def chunk_vec_given_lens(vec, lens, dim=None):
    if dim is None:
        dim = len(vec.shape)
    assert dim in [2, 3]
    csum = list(np.cumsum(lens))
    if dim == 2:
        return [vec[:, start:end] for (start, end) in zip([0] + csum[:-1], csum)]
    elif dim == 3:
        return [vec[:, :, start:end] for (start, end) in zip([0] + csum[:-1], csum)]


class BlockNADE(nn.Module):
    def __init__(self, nade_vis_units, nade_hid_units, nade_input_size):
        super(BlockNADE, self).__init__()

        self.n_blocks = len(nade_vis_units)
        self.visible_units = nade_vis_units  # a list of units per block
        self.hidden_units = nade_hid_units

        # Biases
        self.b_vis_layer = nn.ModuleList(
            [nn.Linear(nade_input_size, n, bias=False) for n in self.visible_units]
        )
        self.b_hid_layer = nn.Linear(nade_input_size, self.hidden_units, bias=False)

        self.HtoV0 = nn.Parameter(torch.randn(self.hidden_units, self.visible_units[0]))
        self.HtoV1 = nn.Parameter(torch.randn(self.hidden_units, self.visible_units[1]))
        self.HtoV2 = nn.Parameter(torch.randn(self.hidden_units, self.visible_units[2]))

        self.VtoH0 = nn.Parameter(torch.randn(self.visible_units[0], self.hidden_units))
        self.VtoH1 = nn.Parameter(torch.randn(self.visible_units[1], self.hidden_units))
        self.VtoH2 = nn.Parameter(torch.randn(self.visible_units[2], self.hidden_units))

        self.b_vis0 = nn.Parameter(torch.zeros(self.visible_units[0]))
        self.b_vis1 = nn.Parameter(torch.zeros(self.visible_units[1]))
        self.b_vis2 = nn.Parameter(torch.zeros(self.visible_units[2]))

        self.b_hid = nn.Parameter(torch.zeros(self.hidden_units))

    def update_biases(self, input_layer):
        self.b = [
            self.b_vis_layer[0](input_layer) + self.b_vis0,
            self.b_vis_layer[1](input_layer) + self.b_vis1,
            self.b_vis_layer[2](input_layer) + self.b_vis2,
        ]
        self.c = self.b_hid_layer(input_layer) + self.b_hid

    def forward(self, input_layer, y=None):
        """input_layer.shape == [batch, timesteps, units], y.shape == [B, T, outputs]"""
        self.update_biases(input_layer)

        if y is not None:
            targets = chunk_vec_given_lens(y, self.visible_units, len(y.shape))
        else:
            targets = None

        h = self.c
        logits = []
        for i in range(self.n_blocks):
            if i == 0:
                x = torch.tensordot(torch.sigmoid(h), self.HtoV0, dims=1) + self.b[i]
            elif i == 1:
                x = torch.tensordot(torch.sigmoid(h), self.HtoV1, dims=1) + self.b[i]
            elif i == 2:
                x = torch.tensordot(torch.sigmoid(h), self.HtoV2, dims=1) + self.b[i]
            if targets is None:
                winner = Categorical(logits=x).sample()
                input_vtoh = nn.functional.one_hot(
                    winner, num_classes=self.visible_units[i]
                ).float()
            else:
                input_vtoh = targets[i]
            logits.append(x)
            if i != self.n_blocks - 1:  # teacher forcing
                if i == 0:
                    h += torch.tensordot(input_vtoh, self.VtoH0, dims=1)
                elif i == 1:
                    h += torch.tensordot(input_vtoh, self.VtoH1, dims=1)
                elif i == 2:
                    h += torch.tensordot(input_vtoh, self.VtoH2, dims=1)
        return torch.cat(logits, dim=2)

    def sample(self, input_layer, temp):
        self.update_biases(input_layer)
        h_i = self.c

        output = []
        logits = []
        for i in range(self.n_blocks):
            if i == 0:
                logits_i = (
                    torch.tensordot(torch.sigmoid(h_i), self.HtoV0, dims=1) + self.b[i]
                )
            elif i == 1:
                logits_i = (
                    torch.tensordot(torch.sigmoid(h_i), self.HtoV1, dims=1) + self.b[i]
                )
            elif i == 2:
                logits_i = (
                    torch.tensordot(torch.sigmoid(h_i), self.HtoV2, dims=1) + self.b[i]
                )
            winner = Categorical(logits=logits_i / temp).sample()
            output_i = nn.functional.one_hot(
                winner, num_classes=self.visible_units[i]
            ).float()
            logits.append(logits_i)
            output.append(output_i)
            if i != self.n_blocks - 1:
                if i == 0:
                    h_i += torch.tensordot(output_i, self.VtoH0, dims=1)
                elif i == 1:
                    h_i += torch.tensordot(output_i, self.VtoH1, dims=1)
        return torch.cat(logits, dim=2), torch.cat(output, dim=2)


class CRF(nn.Module):
    """
    Implements Conditional Random Fields that can be trained via
    backpropagation.
    """

    def __init__(self, num_tags):
        super(CRF, self).__init__()

        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.stop_transitions = nn.Parameter(torch.randn(num_tags))

        nn.init.xavier_normal_(self.transitions)

    def forward(self, feats):
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        return self._viterbi(feats)

    def loss(self, feats, tags):
        """
        Computes negative log likelihood between features and tags.
        Essentially difference between individual sequence scores and
        sum of all possible sequence scores (partition function)
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns:
            Negative log likelihood [a scalar]
        """
        # Shape checks
        if len(feats.shape) != 3:
            raise ValueError("feats must be 3-d got {}-d".format(feats.shape))

        if len(tags.shape) != 2:
            raise ValueError("tags must be 2-d but got {}-d".format(tags.shape))

        if feats.shape[:2] != tags.shape:
            raise ValueError("First two dimensions of feats and tags must match")

        sequence_score = self._sequence_score(feats, tags)
        partition_function = self._partition_function(feats)
        log_probability = sequence_score - partition_function

        # -ve of l()
        # Average across batch
        return -log_probability.mean()

    def _sequence_score(self, feats, tags):
        """
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
            tags: Target tag indices [batch size, sequence length]. Should be between
                    0 and num_tags
        Returns: Sequence score of shape [batch size]
        """

        # Compute feature scores
        feat_score = feats.gather(2, tags.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

        # Compute transition scores
        # Unfold to get [from, to] tag index pairs
        tags_pairs = tags.unfold(1, 2, 1)

        # Use advanced indexing to pull out required transition scores
        indices = tags_pairs.permute(2, 0, 1).chunk(2)
        trans_score = self.transitions[indices].squeeze(0).sum(dim=-1)

        # Compute start and stop scores
        start_score = self.start_transitions[tags[:, 0]]
        stop_score = self.stop_transitions[tags[:, -1]]

        return feat_score + start_score + trans_score + stop_score

    def _partition_function(self, feats):
        """
        Computes the partitition function for CRF using the forward algorithm.
        Basically calculate scores for all possible tag sequences for
        the given feature vector sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns:
            Total scores of shape [batch size]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError(
                "num_tags should be {} but got {}".format(self.num_tags, num_tags)
            )

        a = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(
            0
        )  # [1, num_tags, num_tags] from -> to

        for i in range(1, seq_size):
            feat = feats[:, i].unsqueeze(1)  # [batch_size, 1, num_tags]
            a = self._log_sum_exp(
                a.unsqueeze(-1) + transitions + feat, 1
            )  # [batch_size, num_tags]

        return self._log_sum_exp(
            a + self.stop_transitions.unsqueeze(0), 1
        )  # [batch_size]

    def _viterbi(self, feats):
        """
        Uses Viterbi algorithm to predict the best sequence
        Parameters:
            feats: Input features [batch size, sequence length, number of tags]
        Returns: Best tag sequence [batch size, sequence length]
        """
        _, seq_size, num_tags = feats.shape

        if self.num_tags != num_tags:
            raise ValueError(
                "num_tags should be {} but got {}".format(self.num_tags, num_tags)
            )

        v = feats[:, 0] + self.start_transitions.unsqueeze(0)  # [batch_size, num_tags]
        transitions = self.transitions.unsqueeze(
            0
        )  # [1, num_tags, num_tags] from -> to
        paths = []

        for i in range(1, seq_size):
            feat = feats[:, i]  # [batch_size, num_tags]
            v, idx = (v.unsqueeze(-1) + transitions).max(
                1
            )  # [batch_size, num_tags], [batch_size, num_tags]

            paths.append(idx)
            v = v + feat  # [batch_size, num_tags]

        v, tag = (v + self.stop_transitions.unsqueeze(0)).max(1, True)

        # Backtrack
        tags = [tag]
        for idx in reversed(paths):
            tag = idx.gather(1, tag)
            tags.append(tag)

        tags.reverse()
        return torch.cat(tags, 1)

    def _log_sum_exp(self, logits, dim):
        """
        Computes log-sum-exp in a stable way
        """
        max_val, _ = logits.max(dim)
        return max_val + (logits - max_val.unsqueeze(dim)).exp().sum(dim).log()


class Crf(nn.Module):
    def __init__(self, num_chords, timestep):
        super(Crf, self).__init__()
        self.output_size = num_chords
        self.Crf = CRF(self.output_size)

    def forward(self, probs, labels):
        prediction = self.Crf(probs)
        prediction = prediction.view(-1)
        labels = labels.view(-1, labels.shape[-1])
        loss = self.Crf.loss(probs, labels)
        return prediction, loss
