import math
import torch
import inspect
from torch import Tensor, nn, int32
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange, reduce
from typing import Tuple

from recipes.umm2.models.umm_melrof import MLP_embed, MLP_proj
from recipes.umm2.models.base import BaseStage
from recipes.umm2.modules.utils import get_codebook_distance_stats_batched


class VQ(BaseStage):
    def __init__(
        self, 
        config,
        takes=["latent"], 
        provides=["latent", "vq_ids", "vq_entropy"],
        bypasses=[],
        task="vq",
        loss_weight=1.0,
        lr_ratio=1.0,
        vq_scheme=None,
        is_frozen=False,
        
    ):
        super().__init__(takes, provides, bypasses, task, loss_weight, lr_ratio, is_frozen=is_frozen)

        self.config = config
        self.num_bands = 0  # not using multi-bands
        self.vq_codebook_size = config.vq_codebook_size

        if vq_scheme is None:
            self.vq = EMAVectorQuantizer(
                codebook_size=config.vq_codebook_size,
                codebook_dim=config.vq_codebook_dim,
                decay=config.vq_decay,
            )
        else:
            self.vq = vq_scheme

        if config.get("vq_proj_norm", None) == "melrof":
            self.vq_proj_in = MLP_embed(
                num_feature=config.num_feature, 
                mel_bands=config.mel_bands,
                out_dim=int(config.vq_codebook_dim//config.mel_bands),
                use_checkpoint=config.use_checkpoint
                )
            self.vq_proj_out = MLP_proj(
                in_dim=int(config.vq_codebook_dim//config.mel_bands),
                num_feature=config.num_feature,
                mel_bands=config.mel_bands,
                )
        elif config.get("vq_proj_norm", None) == "bn":
            self.vq_proj_in = nn.Sequential(
                Transpose(),
                WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1)
                if config.hidden_size != config.vq_codebook_dim
                else nn.Identity(),
                nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
                Transpose(),
            )
            self.vq_proj_out = nn.Sequential(
                Transpose(),
                WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity(),
                Transpose(),
            )
        elif config.get("vq_proj_norm", None) == "ln":
            self.vq_proj_in = nn.Sequential(
                nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False)
                if config.hidden_size!= config.vq_codebook_dim
                else nn.Identity(),
                nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
            )
            self.vq_proj_out = nn.Sequential(
                nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False)
                if config.vq_codebook_dim != config.hidden_size
                else nn.Identity()
            )
        else:
            self.vq_proj_in = nn.Linear(
                config.hidden_size, config.vq_codebook_dim, bias=False
            )
            self.vq_proj_out = nn.Linear(
                config.vq_codebook_dim, config.hidden_size, bias=False
            )
        
        if config.get("vq_proj_noise", 0) > 0:
            self.register_buffer("cnt", torch.FloatTensor([0]))

        if is_frozen:
            self.eval()

    def get_nuc(self, target_tokens, attn_mask=None):
        """
        Efficient vectorized version that handles both masked and unmasked cases.
        Uses tensor operations only, avoiding loops and expensive per-sample operations.
        With attn_mask, the memory and computatuion will be 2X compared to without mask
        The good strategy is to : 
        * Get estimated statistics for each step without attn_mask to track trends
        * Get accurate statistics for every N steps 
        
        Args:
            target_tokens: Input tokens, shape (batch, seq_len) or (batch, height, width)
            attn_mask: Optional attention mask, shape should match target_tokens
            
        Returns:
            float: Average normalized unique count across batch
        """
        # Handle input dimensions
        if target_tokens.dim() == 3:
            flat_tokens = target_tokens.view(target_tokens.size(0), -1)
            if attn_mask is not None and attn_mask.dim() == 2:
                # Expand mask to match flattened tokens
                attn_mask = attn_mask.unsqueeze(-1).expand_as(target_tokens).contiguous()
                attn_mask = attn_mask.view(attn_mask.size(0), -1)
        elif target_tokens.dim() == 2:
            flat_tokens = target_tokens
        else:
            raise ValueError(f"Input shape wrong, got {target_tokens.size()}")
        
        batch_size, seq_len = flat_tokens.shape
        device = flat_tokens.device
        
        # Branch based on whether we have a mask
        if attn_mask is None:
            # Unmasked case - use original efficient method
            sorted_tokens, _ = torch.sort(flat_tokens, dim=1)
            
            # Find positions where tokens change
            unique_mask = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                sorted_tokens[:, 1:] != sorted_tokens[:, :-1]
            ], dim=1)
            
            unique_counts = unique_mask.sum(dim=1).float()
            return (unique_counts.mean() / seq_len).item()
        
        else:
            # Masked case - handle attention mask
            # Replace masked positions with a special "ignore" value
            ignore_value = flat_tokens.max().item() + 1
            masked_tokens = torch.where(attn_mask, flat_tokens, ignore_value)
            
            # Sort each sequence
            sorted_tokens, _ = torch.sort(masked_tokens, dim=1)
            
            # Create mask for valid positions (ignoring the special ignore_value)
            is_valid = sorted_tokens != ignore_value
            
            # Find where tokens change (indicating new unique tokens)
            token_changes = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                sorted_tokens[:, 1:] != sorted_tokens[:, :-1]
            ], dim=1)
            
            # Only count changes for valid positions
            valid_unique_mask = token_changes & is_valid
            
            # Count unique tokens per sample
            unique_counts = valid_unique_mask.sum(dim=1).float()
            
            # Get valid lengths per sample and avoid division by zero
            valid_lengths = torch.clamp(attn_mask.sum(dim=1).float(), min=1.0)
            
            # Normalize by valid lengths and return mean
            normalized_counts = unique_counts / valid_lengths
            return normalized_counts.mean().item()



    def forward(self, data):
        hidden_states = data['latent']
        attn_mask = data['attn_mask']

        hidden_states = self.vq_proj_in(hidden_states)
        to_quantize_embs = hidden_states

        if self.config.get("vq_proj_norm", None) == "melrof":
            num_bands = hidden_states.shape[2]
            hidden_states = rearrange(hidden_states, "b t k d -> b t (k d)") # [batch, time, vq_codebook_dim]

        if self.config.get("vq_proj_noise", 0) > 0 and self.training:
            noise_scale = (self.config.vq_proj_noise - self.cnt).clamp(0) / self.config.vq_proj_noise
            hidden_states = (
                hidden_states + torch.randn_like(hidden_states) * noise_scale
            )
            self.cnt.add_(1)
        if "e_scale" in inspect.getfullargspec(self.vq.forward).args:
            e_scale = 0.
            if self.training and hasattr(self, "cnt") and self.cnt < 30000:
                e_scale = 1.0
            vq_output_dict = self.vq(hidden_states, padding_mask=attn_mask, e_scale=e_scale )
        else:
            vq_output_dict = self.vq(hidden_states, padding_mask=attn_mask)

        quantized_embs = vq_output_dict["embs"]

        if self.config.get("vq_proj_norm", None) == "melrof":
            hidden_states = rearrange(hidden_states, "b t (k d) -> b t k d", k=num_bands)

        quantized_out = self.vq_proj_out(quantized_embs)

        estimated_code_rate = self.get_nuc(vq_output_dict["ids"], attn_mask=None)
        
        output_dict = {
            "prevq_embs": to_quantize_embs,
            "quantized_out": quantized_out,
            "vq_ids": vq_output_dict["ids"],
            "vq_embs": quantized_embs,
            "loss": vq_output_dict["loss"],
            f"aux/loss_{self.task}": vq_output_dict["loss"], 
            "aux/vq_entropy": vq_output_dict["entropy"], 
            "aux/estimated_code_rate" : estimated_code_rate,
        } 
    
        if self.cnt % 1000 == 0:
            code_rate = self.get_nuc(vq_output_dict["ids"], attn_mask=attn_mask.bool())
            output_dict['aux/code_rate'] = code_rate

            codebook_dist_stats = get_codebook_distance_stats_batched(self.vq.embedding.weight,batch_size=1024)
            for key, value in codebook_dist_stats.items():
                output_dict[f"aux/{key}"] = value

        return output_dict

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class FeaturePool:
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """

    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1) / pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)
        if self.nums_features < self.pool_size:
            if (
                features.size(0) > self.pool_size
            ):  # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(
                    0, features.size(0), (int(self.pool_size),)
                )
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features : num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(
                    0, features.size(0), (int(self.pool_size),)
                )
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[: features.size(0)]] = features

        return self.features


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class ClusteredVectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim,
        beta=0.25,
        distance="cos",
        anchor="probrandom",
        first_batch=False,
        contras_loss=True,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.codebook_size, self.codebook_dim)
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )
        self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    def forward(self, z):
        z_flattened = rearrange(z, "b t d -> (b t) d")

        # clculate the distance
        if self.distance == "l2":
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = (
                -torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True)
                - torch.sum(self.embedding.weight**2, dim=1)
                + 2
                * torch.einsum(
                    "bd, dn-> bn",
                    z_flattened.detach(),
                    rearrange(self.embedding.weight, "n d-> d n"),
                )
            )
        elif self.distance == "cos":
            # cosine distances from z to embeddings e_j
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum(
                "bd,dn->bn",
                normed_z_flattened,
                rearrange(normed_codebook, "n d -> d n"),
            )

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:, -1]
        encodings = torch.zeros(
            encoding_indices.unsqueeze(1).shape[0], self.codebook_size, device=z.device
        )
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # count
        avg_probs = torch.mean(encodings, dim=0)

        # online clustered reinitialisation for unoptimized points
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            # running average updates
            if self.anchor in ["closest", "random", "probrandom"] and (not self.init):
                # closest sampling
                if self.anchor == "closest":
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1, :]]
                # feature pool based random sampling
                elif self.anchor == "random":
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == "probrandom":
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = (
                    torch.exp(
                        -(self.embed_prob * self.codebook_size * 10) / (1 - self.decay)
                        - 1e-3
                    )
                    .unsqueeze(1)
                    .repeat(1, self.codebook_dim)
                )
                self.embedding.weight.data = (
                    self.embedding.weight.data * (1 - decay) + random_feat * decay
                )
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[
                    -max(1, int(sort_distance.size(0) / self.codebook_size)) :, :
                ].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[: int(sort_distance.size(0) * 1 / 2), :]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(
                    dis,
                    torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device),
                )
                loss += contra_loss

        encoding_indices = rearrange(encoding_indices, "(b t) -> b t", t=z.size(1))

        output_dict = {
            "embs": z_q, 
            "ids": encoding_indices, 
            "loss": loss,
            "entropy": self.entorpy(),
        }
        return output_dict

    @torch.no_grad()
    def entropy(self):
        p = self.embed_prob / self.embed_prob.sum()
        entropy = -torch.sum(p * torch.log(p + 1e-10))
        return entropy


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, codebook_size):
        super().__init__()
        _recommended_levels = {
            256: [8, 6, 5],
            1024: [8, 5, 5, 5],
            4096: [7, 5, 5, 5, 5],
            16384: [8, 8, 8, 6, 5],
            32768: [8, 8, 8, 8, 8],
            65536: [8, 8, 8, 5, 5, 8],
        }
        if codebook_size not in _recommended_levels:
            raise KeyError(
                f"{codebook_size} is not in one of the recommended FiniteScalarQuantizer levels"
            )
        levels = _recommended_levels[codebook_size]
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis)

        self.dim = len(levels)
        self.n_codes = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.n_codes))
        self.register_buffer("implicit_codebook", implicit_codebook)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        zhat = self.quantize(z)
        indices = self.codes_to_indices(zhat)

        output_dict = {
            "embs": zhat, 
            "ids": indices, 
            "loss": None,
            "entropy": None,
        }
        return output_dict

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered)


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()

def binary_entropy(prob):
    return -prob * log(prob) - (1 - prob) * log(1 - prob)

class LookupFreeQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size,
        entropy_loss_weight=0.1,
        diversity_gamma=2.5,
        straight_through_activation=nn.Tanh(),
    ):
        super().__init__()
        assert math.log2(codebook_size).is_integer()
        self.codebook_dim = int(math.log2(codebook_size))
        self.activation = straight_through_activation
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.register_buffer("mask", 2 ** torch.arange(self.codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.zeros(1), persistent=False)

    def indices_to_codes(self, indices):
        # indices to codes, which are bits of either -1 or 1
        bits = ((indices[..., None].int() & self.mask) != 0).float()
        codes = bits * 2 - 1
        return codes

    def forward(self, x, inv_temperature=1.0):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        """
        # quantize by eq 3.

        ones = torch.ones_like(x)
        quantized = torch.where(x > 0, ones, -ones)

        # use straight-through gradients with tanh (or custom activation fn) if training

        if self.training:
            x = self.activation(x * inv_temperature)
            x = x - x.detach() + quantized
        else:
            x = quantized

        # calculate indices

        indices = reduce((x > 0).int() * self.mask.int(), "b n d -> b n", "sum")

        # entropy aux loss

        if self.training:
            prob = (x * inv_temperature).sigmoid()

            bit_entropy = binary_entropy(prob).mean()

            avg_prob = reduce(prob, "b n d -> b d", "mean")
            codebook_entropy = binary_entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each bit, so each scalar commits to one latent binary bit or the other
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used

            entropy_aux_loss = bit_entropy - self.diversity_gamma * codebook_entropy
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = self.zero

        entropy_aux_loss = entropy_aux_loss * self.entropy_loss_weight

        output_dict = {
            "embs": x, 
            "ids": indices, 
            "loss": entropy_aux_loss,
            "entropy": None,
        }
        return output_dict


class EMAVectorQuantizerEntropy(nn.Module):
    def __init__(
        self, 
        codebook_size, 
        codebook_dim, 
        same_index_shape=True, 
        decay=0.99, 
        dist=True,
        distance_type="euclidean"
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.same_index_shape = same_index_shape
        self.dist = dist
        self.distance_type = distance_type
        
        # Validate distance type
        valid_distances = ["cosine", "euclidean", "dot_product"]
        if distance_type not in valid_distances:
            raise ValueError(f"distance_type must be one of {valid_distances}, got {distance_type}")
        
        should_normalize_at_init = (self.distance_type == "cosine")
        self.embedding = EMAEmbedding(
            self.codebook_size, 
            self.codebook_dim, 
            decay=decay, 
            learnable=True,
            normalize_init=should_normalize_at_init
        )

    def _compute_distance(self, x_flat: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between input vectors and codebook vectors.
        
        Args:
            x_flat: Input vectors of shape [N, D]
            codebook: Codebook vectors of shape [K, D]
            
        Returns:
            distances: Distance matrix of shape [N, K]
        """
        if not hasattr(self, "distance_type"):
            self.distance_type = "euclidean"
            
        if self.distance_type == "cosine":
            # Normalize vectors for cosine similarity

            x_norm = F.normalize(x_flat, p=2, dim=1)  # [N, D]
            codebook_norm = F.normalize(codebook, p=2, dim=1)  # [K, D]
            
            # Cosine similarity
            cosine_sim = torch.matmul(x_norm, codebook_norm.t())  # [N, K]
            
            # Convert to distance (1 - cosine_similarity)
            distances = 1.0 - cosine_sim
            
        elif self.distance_type == "euclidean":
            # Squared Euclidean distance
            distances = (
                torch.sum(x_flat**2, dim=1, keepdim=True)  # [N, 1]
                + torch.sum(codebook**2, dim=1)  # [K]
                - 2 * torch.matmul(x_flat, codebook.t())  # [N, K]
            )
            
        elif self.distance_type == "dot_product":
            # Negative dot product (higher dot product = smaller distance)
            distances = -torch.matmul(x_flat, codebook.t())  # [N, K]
            
        return distances

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits for the input tensor."""
        x_flat = x.view(-1, self.codebook_dim)
        
        # Calculate distances using the specified distance function
        distances = self._compute_distance(x_flat, self.embedding.weight)
        
        # Return negative distances as logits (lower distance = higher logit)
        return -distances.view(x.shape[0], x.shape[1], -1)  # (B, T, N)

    def forward(self, z, padding_mask, e_scale=1.0):
        z_flattened = rearrange(z, "b t d -> (b t) d")
        
        # Compute distances using the specified distance function
        d = self._compute_distance(z_flattened, self.embedding.weight)
        
        min_encoding_indices = torch.argmin(d, dim=1)  # [b*h]
        z_q = self.embedding(min_encoding_indices).view(z.shape)  # [b*h, c] -> [b, h, c]
        
        valid_mask = rearrange(padding_mask, "b t -> (b t)")
        valid_indices = torch.where(valid_mask)[0]
        z_flattened_valid = z_flattened[valid_indices]  
        min_encoding_indices_valid = min_encoding_indices[valid_indices]  # [n_valid]
        d_valid = d
        d_valid = d_valid[valid_indices] # inplace modification to avoid copy large d matrix

        # Compute reconstruction loss only on valid positions
        z_q_flattened_valid = self.embedding(min_encoding_indices_valid)  # [n_valid, d]
        reconstruction_loss = torch.mean((z_q_flattened_valid.detach() - z_flattened_valid) ** 2)
        
        # Compute entropy loss only on valid positions
        entropy_loss = e_scale * self.entropy_loss(-d_valid, loss_type="softmax")

        # EMA update
        if self.training and self.embedding.update:
            one_hot_valid = F.one_hot(min_encoding_indices_valid, self.codebook_size).type(
                z.dtype
            )  # [n_valid, k]
            # EMA cluster size
            one_hot_sum = one_hot_valid.sum(0)  # [k]
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
            # EMA embedding average
            embed_sum = (
                one_hot_valid.transpose(0, 1) @ z_flattened_valid
            )  # [k, b*h] * [b*h, c] = [k, c]
            if self.dist:
                torch.distributed.all_reduce(embed_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.codebook_size)

            if self.distance_type == "cosine":
                # Re-normalize the codebook weights to keep them on the unit sphere.
                F.normalize(self.embedding.weight.data, p=2, dim=1, out=self.embedding.weight.data)

        loss = reconstruction_loss + entropy_loss
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        output_dict = {
            "embs": z_q, 
            "ids": min_encoding_indices, 
            "loss": loss,
            "entropy": self.entropy(),
        }
        return output_dict

    def entropy_loss(self, affinity, loss_type="softmax", temperature=0.7, eps=1e-10):
        # affinity: [b, t, d_n]
        flat_affinity = affinity.reshape(
            -1, affinity.shape[-1]
        )  # [b, t, d_n] -> [b*t, d_n]
        flat_affinity = flat_affinity / temperature
        probs = flat_affinity.softmax(dim=-1)  # [b*t, d_n]
        log_probs = (probs + eps).log()  # [b*t, d_n]
        if loss_type == "softmax":
            target_probs = probs
        elif loss_type == "argmax":
            codes = flat_affinity.argmax(dim=-1)  # [b*t]
            onehots = F.one_hot(codes, flat_affinity.shape[-1]).float()  # [b*t, d_n]
            onehots = probs - (probs - onehots).detach()  # [b*t, d_n]
            target_probs = onehots  # [b*t, d_n]
        else:
            raise ValueError("Entropy loss {} not supported".format(loss_type))
        avg_probs = torch.mean(target_probs, dim=0)  # [b*t, d_n] -> [d_n]
        avg_entropy = -torch.sum(avg_probs * (avg_probs + eps).log())  # [d_n] -> []
        sample_entropy = -torch.mean(
            torch.sum(target_probs * log_probs, dim=-1)
        )  # [b*t, d_n] * [b*t, d_n] -> [b*t] -> []
        loss = 0.1 * (sample_entropy - 2.5 * avg_entropy)

        return loss

    def entropy(self):
        return self.embedding.entropy()





class EMAEmbedding(nn.Module):
    def __init__(
        self,
        codebook_size,
        codebook_dim,
        decay=0.99,
        eps=1e-5,
        learnable=False,
        orthonormal_init=False,
        normalize_init=False
    ):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.learnable = learnable

        weight = torch.randn(codebook_size, codebook_dim, dtype=torch.float32)
        if orthonormal_init:
            weight = torch.qr(weight)[0]
            std = weight.std(dim=1).unsqueeze(1)
            weight = weight / std

        # For cosine distance
        if normalize_init:
            weight = F.normalize(weight, dim=1)

        weight[0] = 0.0
        if not learnable:
            self.register_buffer("weight", weight)
        else:
            self.register_parameter("weight", nn.Parameter(weight))
        self.register_buffer("cluster_size", torch.zeros(codebook_size) + 8)
        self.register_buffer("embed_avg", weight.clone())
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size.data, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(
            new_embed_avg.data, alpha=1 - self.decay
        )

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized.data)
        if not self.learnable:
            # make sure the greedy algorithm to get best estimation
            self.weight.data[0] = 0.0
            self.embed_avg[0] = 0.0

    @torch.no_grad()
    def entropy(self):
        p = self.cluster_size / self.cluster_size.sum()
        p = p.clamp(1e-9)
        entropy = (-p * p.log()).sum()
        return entropy

    @torch.no_grad()
    def remap_weight(self, thres=2):
        cs = self.cluster_size.data
        avg_w = self.embed_avg.data
        w = self.weight.data
        # ranking
        topk_res = torch.topk(cs[1:], k=cs.shape[-1] - 1)  # 大到小
        indices = topk_res.indices + 1
        # remap
        new_cs = torch.zeros_like(cs)
        new_avg_w = torch.zeros_like(avg_w)
        new_w = torch.zeros_like(w)
        cnt = 0
        for j, ind in enumerate(indices):
            cs_tmp = cs[ind]
            if cs_tmp < thres:
                print(cs_tmp, j)
                new_w[j + 1] = w[indices[cnt]]
                new_avg_w[j + 1] = w[indices[cnt]] * cs[indices[cnt]] / 2
                new_cs[j + 1] = cs[indices[cnt]] / 2
                cnt += 1
            else:
                new_w[j + 1] = w[ind]
                new_avg_w[j + 1] = avg_w[ind]
                new_cs[j + 1] = cs[ind]
        if cnt == 0:  # tried to fix, "i == 0" --> "cnt == 0"
            new_cs[0] = 8
        else:
            new_cs[0] = cs[0]
        self.cluster_size.data.copy_(new_cs.data)
        self.embed_avg.data.copy_(new_avg_w.data)
        self.weight.data.copy_(new_w.data)


class EMAVectorQuantizer(nn.Module):
    def __init__(
        self, codebook_size, codebook_dim, same_index_shape=True, decay=0.99, dist=True
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay

        # ema for dist
        self.dist = dist
        self.embedding = EMAEmbedding(
            self.codebook_size, self.codebook_dim, decay=decay, learnable=False
        )
        self.same_index_shape = same_index_shape

    def forward(self, z):
        z_flattened = rearrange(z, "b t d -> (b t) d")

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # [b*h]
        z_q = self.embedding(min_encoding_indices).view(
            z.shape
        )  # [b*h, c] -> [b, h, c]

        # EMA updating, use for
        if self.training and self.embedding.update:
            one_hot = F.one_hot(min_encoding_indices, self.codebook_size).type(
                z.dtype
            )  # [b*h, k]
            # EMA cluster size
            one_hot_sum = one_hot.sum(0)  # [k]
            if self.dist:
                torch.distributed.all_reduce(one_hot_sum)
            self.embedding.cluster_size_ema_update(one_hot_sum)
            # EMA embedding average
            embed_sum = (
                one_hot.transpose(0, 1) @ z_flattened
            )  # [k, b*h] * [b*h, c] = [k, c]
            if self.dist:
                torch.distributed.all_reduce(embed_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.codebook_size)

        loss = torch.mean((z_q.detach() - z) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.same_index_shape:
            min_encoding_indices = rearrange(
                min_encoding_indices, "(b t) -> b t", t=z.size(1)
            )

        output_dict = {
            "embs": z_q, 
            "ids": min_encoding_indices, 
            "loss": loss,
            "entropy": self.entropy(),
        }
        return output_dict

    @torch.no_grad()
    def entropy(self):
        return self.embedding.entropy()
