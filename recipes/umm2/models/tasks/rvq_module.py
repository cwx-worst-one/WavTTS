import math
import torch
import inspect
from torch import Tensor, nn, int32
from torch.nn import functional as F
from einops import rearrange, reduce
from recipes.umm2.models.tasks.vq_module import VQ, EMAVectorQuantizerEntropy, EMAVectorQuantizer, EMAEmbedding
from typing import List, Union

class RVQ(VQ):
    def __init__(
        self, 
        config,
        takes=["latent"], 
        provides=["latent", "vq_ids", "vq_entropy"],
        bypasses=[],
        task="rvq",
        loss_weight=1.0,
        lr_ratio=1.0,
        rvq_scheme=None,
        is_frozen=False,
    ):
        super().__init__(
            config, takes, provides, bypasses, task, loss_weight, lr_ratio, 'none', is_frozen
        )

        if rvq_scheme is None:
            self.rvq = EMAResidualVectorQuantizerEntropy(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
                rvq=config.rvq,
            )
        else:
            self.rvq = rvq_scheme


    def forward(self, data):
        hidden_states = data['latent']
        attn_mask = data['attn_mask']

        hidden_states = self.vq_proj_in(hidden_states)
        to_quantize_embs = hidden_states

        # do we really need vq_proj_noise?
        if self.config.get("vq_proj_noise", 0) > 0:
            noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(0) / self.config.vq_proj_noise
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            self.cnt.add_(1)

        # vq_output_dict need to support hidden from each rvq layer 
        # the first rvq layer can directly connnect with CTC
        if "e_scale" in inspect.getfullargspec(self.rvq.forward).args:
            e_scale = 0.
            if self.training and hasattr(self, "cnt") and self.cnt < 30000:
                e_scale = 1.0
            vq_output_dict = self.rvq(
                hidden_states, attn_mask,e_scale=e_scale)
        else:
            vq_output_dict = self.rvq(hidden_states, attn_mask)

        vq_emb = vq_output_dict["embs"]
        vq_ids = vq_output_dict["ids"]

        if self.training:
            batch_size = vq_emb.shape[0]
            # here, quantizer dropout is applied to each [sample]
            if torch.rand(1) > 0.5:
                hidden_states = self.vq_proj_out(vq_emb[..., -1])
            else:
                quantizer_dropout = torch.randint(vq_emb.shape[-1], size=(batch_size,))
                dropout_rvq_embs = vq_emb[range(batch_size), ..., quantizer_dropout]
                hidden_states = self.vq_proj_out(dropout_rvq_embs)
        else:
            if "inference_R" in data:
                R_idx = min(max(data["inference_R"]-1, 0), vq_emb.shape[-1]-1)
                hidden_states = self.vq_proj_out(vq_emb[..., R_idx])
                vq_ids = vq_ids[..., :R_idx+1]
            else:
                hidden_states = self.vq_proj_out(vq_emb[..., -1])

        loss = vq_output_dict["loss"].sum()
        loss_weighted = loss * self.loss_weight

        output_dict = {
            "prevq_embs": to_quantize_embs,
            "quantized_out": hidden_states,
            "vq_ids": vq_ids,
            "vq_emb": vq_emb,
            "loss": loss_weighted,
            f"aux/loss_{self.task}": loss, 

        }

        if "entropy" in vq_output_dict.keys():
            for i, temp_v in enumerate(vq_output_dict["entropy"]):
                output_dict[f"aux/vq_entropy_{i}"] = temp_v
        if "ppl" in vq_output_dict.keys():
            for i, temp_v in enumerate(vq_output_dict["ppl"]):
                output_dict[f"aux/vq_ppl_{i}"] = temp_v      
        # log loss of each layer for check (should be decreasing trend for RVQ)
        for i, temp_v in enumerate(vq_output_dict["loss"]):
            output_dict[f"aux/vq_loss_{i}"] = temp_v.item()
            
        return output_dict

class EMAResidualVectorQuantizerEntropy(nn.Module):
    def __init__(self, 
                 codebook_size: Union[int, List[int]], 
                 codebook_dim: Union[int, List[int]], 
                 same_index_shape: bool = True, 
                 decay: float = 0.99, 
                 dist: bool = True, 
                 distance_type="euclidean", #support "cosine", "euclidean", "dot_product"
                 rvq: int = 1):
        super().__init__()
        self.rvq = rvq
        self.eps = 1e-5
        
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * self.rvq
            
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim] * self.rvq

        self.RVQ = nn.ModuleList([])

        for r in range(self.rvq):
            self.RVQ.append(
                EMAVectorQuantizerEntropy(
                    codebook_size=codebook_size[r], 
                    codebook_dim=codebook_dim[r],
                    same_index_shape=same_index_shape,
                    decay=decay, 
                    dist=dist, 
                    distance_type=distance_type)
            )

    def forward(self, z, padding_mask, e_scale=1.0):
        quantized, indices, rvq_loss, entropy = [], [], [], []
        residual = z
        for r in range(len(self.RVQ)):
            r_output_dict = self.RVQ[r](residual, padding_mask, e_scale=e_scale)
            this_z_q = r_output_dict["embs"]
            residual = residual - this_z_q
            if r == 0:
                quantized.append(this_z_q)
            else:
                quantized.append(quantized[-1] + this_z_q)
            indices.append(r_output_dict["ids"])
            rvq_loss.append(r_output_dict["loss"])
            entropy.append(r_output_dict["entropy"])
        
        # straight-through estimator
        # quantized = (quantized - z_unitnorm.unsqueeze(-1)).detach() + z_unitnorm.unsqueeze(-1)
        output_dict = {
            "embs": torch.stack(quantized, -1),
            "ids": torch.stack(indices, -1),
            "loss": torch.stack(rvq_loss, -1),
            "entropy": torch.stack(entropy, -1),
        }
        return output_dict



# RP means random replace
class EMAEmbeddingRP(EMAEmbedding):
    def __init__(
        self,
        r,
        codebook_size,
        codebook_dim,
        decay=0.99,
        eps=1e-5,
        learnable=False,
        orthonormal_init=False,
    ):
        super().__init__(codebook_size, codebook_dim, decay, eps, learnable, orthonormal_init)
        self.decay = decay
        self.eps = eps
        self.learnable = learnable
        self.r = r

        if r == 0:
            weight = torch.empty(codebook_size, codebook_dim, dtype=torch.float32).normal_() 
            weight = weight / (weight.pow(2).sum(-1, keepdim=True) + self.eps).sqrt()   # unit-norm codebook vector
        else:
            weight = torch.empty(codebook_size, codebook_dim, dtype=torch.float32).normal_() / 10**r
            weight[0] = 0.0 # always 0 vector

        if orthonormal_init:
            weight = torch.qr(weight)[0]
            std = weight.std(dim=1).unsqueeze(1)
            weight = weight / std
        
        if not learnable:
            self.register_buffer("weight", weight)
        else:
            self.register_parameter("weight", nn.Parameter(weight))
        self.register_buffer("cluster_size", torch.zeros(codebook_size) + 8)
        self.register_buffer("embed_avg", weight.clone())
        self.update = True

class EMAVectorQuantizerRP(EMAVectorQuantizerEntropy):
    def __init__(self, r, stale_tolerance=100, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.embedding = EMAEmbeddingRP(
            r, self.codebook_size, self.codebook_dim, decay=self.decay, learnable=False
        )
        self.register_buffer("stale_counter", torch.zeros(self.codebook_size))
        self.stale_tolerance = stale_tolerance
        self.r = r
        self.eps = 1e-5

    def forward(self, z, padding_mask):
        """Notation:
        B: batch size
        T: n_frame
        D: codebook dimension
        N: codebook num
        """
        z_flattened = rearrange(z.detach(), "b t d -> (b t) d") # [B*T, D]

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )   # [B*T, N]

        min_encoding_indices = torch.argmin(d, dim=1)  # [B*T]
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)  # [B*T, D] -> [B, T, D]

        encodings = F.one_hot(min_encoding_indices, self.codebook_size).type(z.dtype)  # [B*T, N]
        # EMA cluster size
        encoding_sum = encodings.sum(0)  # [N]
        if self.dist and self.training:
            torch.distributed.all_reduce(encoding_sum)

        # compute ppl
        avg_probs = encoding_sum.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-5), -1)).mean()

        if self.training and self.embedding.update:
            # =============== EMA Update =============== #
            # update ema_count
            self.embedding.cluster_size = self.embedding.decay * self.embedding.cluster_size + (1 - self.embedding.decay) * encoding_sum  # [N]
            # update embedding
            update_direction = encodings.T.mm(z_flattened)  # [N, D]
            if self.dist:
                torch.distributed.all_reduce(update_direction)
            self.embedding.embed_avg = self.embedding.decay * self.embedding.embed_avg + (1 - self.embedding.decay) * update_direction # [N, D]

            # Laplace smoothing on the ema_counters
            # make sure the denominator will never be zero
            n = torch.sum(self.embedding.cluster_size, dim=-1, keepdim=True)  # 1
            self.embedding.cluster_size = (self.embedding.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n  # [N]
            # update codebook
            self.embedding.weight = self.embedding.embed_avg / self.embedding.cluster_size.unsqueeze(-1) # [N, D]

            # =============== Stale code replace ============= #
            # calculate code usage
            stale_codes = (encoding_sum == 0).float()  # [N]
            self.stale_counter = self.stale_counter * stale_codes + stale_codes

            # random replace codes that haven't been used for a while
            replace_code = (self.stale_counter == self.stale_tolerance).float() # [N]
            if replace_code.max() > 0:
                random_input_idx = torch.randperm(z_flattened.shape[0]) # [B*T, D] => [B*T]
                random_input = z_flattened[random_input_idx].reshape(z_flattened.shape) # [B*T, D]
                if random_input.shape[0] < self.codebook_size:
                    random_input = torch.cat([random_input]*(self.codebook_size // random_input.shape[0] + 1), 0)
                random_input = random_input[:self.codebook_size].contiguous()  # [N, D]
                if self.dist:
                    torch.distributed.broadcast(random_input, 0)

                # random_input: [N, D], replace_code: [N] (bool)
                self.embedding.cluster_size = self.embedding.cluster_size * (1 - replace_code)
                self.stale_counter = self.stale_counter * (1 - replace_code)
                replace_code = replace_code.unsqueeze(-1)
                self.embedding.weight = self.embedding.weight * (1 - replace_code) + random_input * replace_code
                self.embedding.embed_avg = self.embedding.embed_avg * (1 - replace_code) + random_input * replace_code
                
            if self.r == 0:
                # unit-norm codebooks
                self.embedding.weight = self.embedding.weight / (self.embedding.weight.pow(2).sum(-1, keepdim=True) + self.eps).sqrt()
                self.embedding.embed_avg = self.embedding.embed_avg / (self.embedding.embed_avg.pow(2).sum(-1, keepdim=True) + self.eps).sqrt()
            else:
                # always contain an all-zero embedding for residual layers
                self.embedding.weight.data[0] = 0.0
                self.embedding.embed_avg[0] = 0.0
            
        loss = torch.mean((z_q.detach() - z) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        return z_q, min_encoding_indices, loss, perplexity


class EMAVectorQuantizerRPSimple(EMAVectorQuantizerRP):
    def __init__(self, *args, **kwargs,):
        super().__init__(*args, **kwargs)

    def forward(self, z, padding_mask):
        """Notation:
        B: batch size
        T: n_frame
        D: codebook dimension
        N: codebook num
        """
        z_flattened = rearrange(z.detach(), "b t d -> (b t) d") # [B*T, D]

        d = self._compute_distance(z_flattened, self.embedding.weight)
        # print("distance", d)

        min_encoding_indices = torch.argmin(d, dim=1)  # [B*T]
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)  # [B*T, D] -> [B, T, D]
        valid_mask = rearrange(padding_mask, "b t -> (b t)")
        valid_indices = torch.where(valid_mask)[0]
        min_encoding_indices_valid = min_encoding_indices[valid_indices]  # [n_valid]

        z_flattened_valid = z_flattened[valid_indices]  # [n_valid, D]
        z_q_flattened_valid = self.embedding(min_encoding_indices_valid)  # [n_valid, D]
        d_valid = d
        d_valid = d_valid[valid_indices] # inplace modification to avoid copy large d matrix

        encodings = F.one_hot(min_encoding_indices_valid, self.codebook_size).type(z.dtype)  # [B*T, N]
        # EMA cluster size
        encoding_sum = encodings.sum(0)  # [N]
        if self.dist and self.training:
            torch.distributed.all_reduce(encoding_sum)

        # compute ppl
        avg_probs = encodings.float().mean(0)   # [N]
        # print("avg_probs", avg_probs)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps), -1)).mean()
        # print("perplexity", perplexity)

        if self.training and self.embedding.update:
            update_direction = torch.einsum('bk, bn -> kn', encodings, z_flattened_valid)  # N, D
            if self.dist:
                torch.distributed.all_reduce(update_direction)
            current_center = update_direction / (encoding_sum.unsqueeze(-1) + self.eps)

            # EMA update on the codebook 
            # only update codes that have been used
            used_codes = (encoding_sum > 0)  # N
            self.embedding.weight[used_codes] = self.embedding.decay * self.embedding.weight[used_codes] \
                                            + (1 - self.embedding.decay) * current_center[used_codes]  # num_code, N

            # random replace codes that haven't been used for a while
            unused_codes = (encoding_sum == 0)  # N
            self.stale_counter = self.stale_counter * unused_codes + unused_codes
            replace_code = (self.stale_counter == self.stale_tolerance) # N
            if replace_code.sum() > 0:
                random_input_idx = torch.randperm(z_flattened_valid.shape[0])
                random_input = z_flattened_valid[random_input_idx].reshape(z_flattened_valid.shape)
                if random_input.shape[0] < replace_code.sum():
                    random_input = torch.cat([random_input]*(replace_code.sum() // random_input.shape[0] + 1), 0)
                random_input = random_input[:replace_code.sum()]  # num_code, N
                if self.dist:
                    torch.distributed.broadcast(random_input, 0)
                self.embedding.weight[replace_code] = random_input
                self.stale_counter[replace_code] = 0

            if self.r == 0:
                # unit-norm codebooks
                self.embedding.weight = self.embedding.weight / (self.embedding.weight.pow(2).sum(-1, keepdim=True) + self.eps).sqrt()
            else:
                # always contain an all-zero embedding for residual layers
                self.embedding.weight.data[0] = 0.0
            
        loss = torch.mean((z_q_flattened_valid.detach() - z_flattened_valid) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        return z_q, min_encoding_indices, loss, perplexity


class EMAResidualVectorQuantizerRP(EMAResidualVectorQuantizerEntropy):
    def __init__(self, vq_type, codebook_size, codebook_dim, 
                 same_index_shape=True,
                 decay=0.99, dist=True, rvq=1, stale_tolerance=100):
        # fix distance type as "euclidean", as the input embedding-to-be-quantized is unit-normed by default
        super().__init__(codebook_size, codebook_dim, same_index_shape, decay, 
                         dist=dist, rvq=rvq, distance_type='euclidean')
        self.rvq = rvq
        self.eps = 1e-5
        self.RVQ = nn.ModuleList([])
        self.stale_tolerance = stale_tolerance
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * self.rvq
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim] * self.rvq
        for r in range(self.rvq):
            if vq_type == "EMARP":
                self.RVQ.append(
                    EMAVectorQuantizerRP(r=r, stale_tolerance=self.stale_tolerance,
                    codebook_size=codebook_size[r], codebook_dim=codebook_dim[r],
                    same_index_shape=same_index_shape, decay=decay, dist=dist)
                )
            elif vq_type == "EMARPSimple":
                self.RVQ.append(
                    EMAVectorQuantizerRPSimple(r=r, stale_tolerance=self.stale_tolerance,
                    codebook_size=codebook_size[r], codebook_dim=codebook_dim[r],
                    same_index_shape=same_index_shape, decay=decay, dist=dist)
                )
            else:
                raise NotImplementedError(f"vq_type {vq_type} not supported")

    def forward(self, z, padding_mask):
        z_unitnorm = z * torch.rsqrt(z.pow(2).sum(-1, keepdim=True) + self.eps) # [B, T, D]

        quantized = []
        indices = []
        rvq_loss = []
        ppl = []
        residual = z_unitnorm
        for r in range(len(self.RVQ)):
            this_z_q, this_indices, this_vq_loss, this_ppl = self.RVQ[r](residual, padding_mask)
            residual = residual - this_z_q
            if r == 0:
                quantized.append(this_z_q)
            else:
                quantized.append(quantized[-1] + this_z_q)
            indices.append(this_indices)
            rvq_loss.append(this_vq_loss)
            ppl.append(this_ppl)
            # print(r, this_vq_loss)
        
        quantized = torch.stack(quantized, -1)
        indices = torch.stack(indices, -1)
        rvq_loss = torch.stack(rvq_loss, -1)
        ppl = torch.stack(ppl, -1)
        # straight-through estimator (have been called in each VQ)
        # quantized = (quantized - z_unitnorm.unsqueeze(-1)).detach() + z_unitnorm.unsqueeze(-1)

        output_dict = {
            "embs": quantized,
            "ids": indices,
            "loss": rvq_loss,
            "ppl": ppl,
        }
        return output_dict



if __name__ == "__main__":
    args = {
        'r': 0,
        'stale_tolerance': 100,
        'codebook_size': 16384,
        'codebook_dim': 32,
        'decay': 0.99,
        'dist': False,
        # 'learnable': False,
    }
    
    # initialize with same embedding weight
    embedding = EMAEmbeddingRP(r=0, codebook_dim=16384, codebook_size=32, decay=0.99, dist=False)
    
    # original_vq = EMAVectorQuantizerRPSimple(**args)
    # new_vq = EMAVectorQuantizerRPSimplePadding(**args)
    # original_vq.embedding.weight.data.copy_(embedding.weight.T.data)
    # new_vq.embedding.weight.data.copy_(embedding.weight.T.data)

    z = torch.randn(2, 100, 32)
    z = F.normalize(z, p=2, dim=-1)
    padding_mask = torch.ones(2, 100).long()
    padding_mask[0,90:] = 0
    padding_mask[1,40:] = 0

    # z_q, min_encoding_indices, loss, perplexity = original_vq(z)
    # z_q_new, min_encoding_indices_new, loss_new, perplexity_new = new_vq(z, padding_mask)

    new_rvq = EMAResidualVectorQuantizerRP(vq_type="EMARPSimple",
                                           codebook_size=16384,
                                           codebook_dim=32,
                                           decay=0.99,
                                           dist=False,
                                           rvq=4,
                                           stale_tolerance=100,)
    returns = new_rvq(z, padding_mask)
    print(returns)
