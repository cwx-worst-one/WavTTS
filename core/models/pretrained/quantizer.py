""" quantizer"""


from itertools import product
import math
import torch
from torch import nn
import torch.nn.functional as F


class GumbelVectorQuantizer(nn.Module):
    """GumbelVectorQuantizer"""

    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
    ):
        """Vector quantization using gumbel softmax
        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements:
                  (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module).
                        this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to
                               project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1.
                                scales the inner dimensionality of projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.vars = nn.Parameter(torch.FloatTensor(1, num_groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        assert len(temp) == 3, temp

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

    def set_num_updates(self, num_updates):
        """set_num_updates"""
        self.curr_temp = max(self.max_temp * self.temp_decay**num_updates, self.min_temp)

    def get_codebook_indices(self):
        """get_codebook_indices"""
        if self.codebook_indices is None:
            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(self.num_vars**self.groups, -1)
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        """codebook"""
        indices = self.get_codebook_indices()
        return self.vars.squeeze(0).index_select(0, indices).view(self.num_vars**self.groups, -1)

    def sample_from_codebook(self, b, n):
        """sample_from_codebook"""
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert n < cb_size, f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        """to_codebook_index"""
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars**exponent)
        return res

    def forward_idx(self, x):
        """forward_idx"""
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):
        """forward"""
        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz * self.groups, -1)

        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        avg_probs = torch.softmax(x.view(bsz * tsz, self.groups, -1).float(), dim=-1).mean(dim=0)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        var = self.vars
        if self.combine_groups:
            var = var.repeat(1, self.groups, 1)

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        x = x.reshape(bsz * tsz, self.groups, self.num_vars).permute(1, 0, 2)
        var = var.reshape(self.groups, self.num_vars, -1)
        x = torch.bmm(x, var).permute(1, 0, 2).reshape(bsz, tsz, -1)

        if not self.time_first:
            x = x.transpose(1, 2)  # BTC -> BCT

        result["x"] = x

        return result


class RandomProjectionQuantizer(nn.Module):
    """RandomProjectionQuantizer"""

    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        quantizer_num=1,
        prototype_initialization_method="gaussian",
        projection_initialization_method="xavier",
    ):
        """A quantizer based on random projection
        See: https://arxiv.org/pdf/2202.01855.pdf
        Args:
            dim: input dimension (channels)
            codebook_size: the number of code in the codebook
            codebook_dim: the dimension of the the code
            quantizer_num: the number of quantizers.
                    See multi-softmax in https://arxiv.org/abs/2303.01037
            initialization_type: the initialization method of the projection matrix
        """
        super().__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_num = quantizer_num
        self.prototype_initialization_method = prototype_initialization_method
        self.projection_initialization_method = projection_initialization_method

        self.register_buffer(
            "prototypes",
            torch.zeros(
                self.quantizer_num,
                1,
                self.codebook_size,
                self.codebook_dim,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "proj",
            torch.zeros(
                self.input_dim,
                self.quantizer_num * self.codebook_dim,
                requires_grad=False,
            ),
        )
        self._initialize()

    def _initialize(self):
        """Initialize the parameters."""
        if self.prototype_initialization_method == "gaussian":
            nn.init.normal_(self.prototypes)
            F.normalize(self.prototypes, dim=-1, out=self.prototypes)
        else:
            raise ValueError(
                "Unknown prototype_initialization_method:"
                " {}".format(self.prototype_initialization_method)
            )

        if self.projection_initialization_method == "xavier":
            fan_in = self.input_dim
            fan_out = self.codebook_dim
            gain = 1.0
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            with torch.no_grad():
                self.proj.normal_(0, std)
        else:
            raise ValueError(
                "Unknown projection_initialization_method:"
                " {}".format(self.projection_initialization_method)
            )

    def forward(self, x):
        """
        Forward a batch of representations to get discrete codes.
        Args:
            x: [batch_size, dim]
        Returns:
            codes: [batch_size, quantizer_num], torch.int64
        """
        assert len(x.shape) == 2
        batch_size, _ = x.shape

        projected = torch.matmul(x, self.proj)  # [batch_size, quantizer_num*codebook_dim]
        projected = F.normalize(
            projected.view(batch_size, self.quantizer_num, self.codebook_dim),
            p=2,
            dim=-1,
        )  # [batch_size, quantizer_num, codebook_dim]
        projected = projected.permute(1, 0, 2).view(
            self.quantizer_num, batch_size, 1, self.codebook_dim
        )  # [quantizer_num, batch_size, 1, codebook_dim]

        # self.prototypes: [quantizer_num, 1, codebook_size, codebook_dim]
        # it is normalized in the function _initialize

        # TODO: configure multiple distances, such as cosine similarity
        # distances = torch.norm(projected - self.prototypes, p=2, dim=-1)
        # [quantizer_num, batch_size, codebook_size]

        # Save spaces.
        distances = (
            projected.view(self.quantizer_num, batch_size, self.codebook_dim)
            .pow(2)
            .sum(-1, keepdim=True)  # [quantizer_num, batch_size, 1]
            + self.prototypes.view(self.quantizer_num, self.codebook_size, self.codebook_dim)
            .pow(2)
            .sum(-1, keepdim=True)
            .transpose(1, 2)  # [quantizer_num, 1, codebook_size]
            - 2
            * torch.bmm(
                projected.view(self.quantizer_num, batch_size, self.codebook_dim),
                self.prototypes.view(
                    self.quantizer_num, self.codebook_size, self.codebook_dim
                ).transpose(1, 2),
            )  # [quantizer_num, batch_size, codebook_size]
        )

        codes = torch.argmin(distances, dim=-1).transpose(0, 1)
        # [batch_size, quantizer_num]
        return codes
