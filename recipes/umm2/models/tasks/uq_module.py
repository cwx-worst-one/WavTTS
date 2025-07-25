import torch.nn as nn
import torch
import math
from einops import rearrange

class UniversalQuantizer(nn.Module):
    def __init__(self, bits=3, eps=1e-5):
        super().__init__()
        
        self.bits = bits
        self.nbins = 2 ** bits
        self.eps = eps

        self.interval_center = torch.linspace(-1+1./self.nbins, 1-1./self.nbins, self.nbins).reshape(1, -1)
    
    @torch.no_grad()
    def round(self, x):
        x_shape = x.shape
        x = x.reshape(-1, 1)
        interval_center = self.interval_center.to(x)
        indices = torch.argmin(torch.abs(x - interval_center), -1)
        x = interval_center[:,indices].reshape(x_shape)
        return x, indices.reshape(x_shape)
    
    def noisy_round(self, x):
        u = torch.empty_like(x).uniform_(-1 + self.eps, 1 - self.eps) / self.nbins
        x_round, indices = self.round(x + u)
        x = (x_round - x).detach() + x - u
        return x, indices
    
    def int2binary(self, x, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(x.dtype)
    
    def binary2int(self, x):
        num_bits = x.shape[-1]
        # Create a tensor of powers of 2
        powers = 2 ** torch.arange(num_bits - 1, -1, -1).to(x.device)
        # Multiply the binary tensor by the powers of 2 and sum along the last dimension
        int_tensor = (x * powers).sum(dim=-1)
        return int_tensor
    
    def emb2index(self, x):
        # x shape: *, N
        import pdb; pdb.set_trace()
        x_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        # get UQ index
        _, indices = self.round(x)  # *, N
        # UQ index to binary
        indices = self.int2binary(indices, self.bits)  # *, N, K
        indices = rearrange(indices, 'b n k -> b (n k)')
        # binary merge to a single index
        indices = self.binary2int(indices).reshape(x_shape[:-1])  # *
        return indices

    def index2emb(self, indices, x_dim):
        # indices shape: *
        indices_shape = indices.shape
        indices = indices.reshape(-1,)
        # single index to binary
        indices = self.int2binary(indices, x_dim*self.bits)  # *, K
        # split binary to UQ dimension
        indices = rearrange(indices, 'b (n k) -> (b n) k', k=self.bits)
        # reform UQ indices
        indices = self.binary2int(indices)  # *
        # UQ indices to UQ emb
        interval_center = self.interval_center.to(indices.device)
        x = interval_center[:,indices].reshape(indices_shape+(-1,))
        return x
    
    def forward(self, x, return_indices=False):
        if self.training:
            # add uniform noise and round to interval center
            x, _ = self.noisy_round(x)
        else:
            # directly round to interval center
            x, _ = self.round(x)
        
        if not return_indices:
            return x
        
        indices = self.emb2index(x)
        return x, indices


class HierarchicalUniversalQuantizer(nn.Module):
    def __init__(self, huq=2, bit=2, eps=1e-5):
        super().__init__()
        self.h = huq
        self.HUQ = nn.ModuleList([])
        for h in range(huq):
            self.HUQ.append(UniversalQuantizer(bit=bit, eps=eps))

    def uq_dim(self, dim):
        assert dim % self.h == 0, \
            f"Dimension {dim} is not divisible by HUQ {self.h}"
        uq_dim = int(dim // self.h)
        return uq_dim
    
    def forward(self, z):
        """
        input: 
            z: [B, T, D]
        output: 
            z_q: [B, T, D]
            huq_id: [B, T, D]    
        """
        dim = z.shape[-1]
        uq_dim = self.uq_dim(dim)
        z_q = []
        huq_id = []
        for h in range(self.h):
            this_zq, this_uq_id = self.HUQ[h](z[..., h*uq_dim:(h+1)*uq_dim])
            z_q.append(this_zq)
            huq_id.append(this_uq_id)
        z_q = torch.cat(z_q, dim=-1)
        huq_id = torch.cat(huq_id, dim=-1)
        
        return z_q, huq_id