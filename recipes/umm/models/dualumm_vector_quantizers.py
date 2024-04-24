from torch import nn
import torch

from recipes.umm.models.umm_mkii import (
    ClusteredVectorQuantizer,
    EMAVectorQuantizerEntropy,
    FiniteScalarQuantizer,
    LookupFreeQuantizer,
    Transpose,
    WNConv1d,
)
from recipes.umm.models.vq import EMAVectorQuantizer


def get_vector_quantizer(vq_type, config):
    """Select VQ, otherwise default to Exponential Moving Average (EMA) VQ."""
    if vq_type == "CVQ":
        return ClusteredVectorQuantizer(
            codebook_size=config.vq_codebook_size,
            codebook_dim=config.vq_codebook_dim,
            distance=config.get("vq_distance", "cos"),
        )
    elif vq_type == "FSQ":
        return FiniteScalarQuantizer(codebook_size=config.vq_codebook_size)
    elif vq_type == "LFQ":
        return LookupFreeQuantizer(codebook_size=config.vq_codebook_size)
    elif vq_type == "EMAEntropy":
        return EMAVectorQuantizerEntropy(
            codebook_size=config.vq_codebook_size,
            codebook_dim=config.vq_codebook_dim,
            decay=config.vq_decay,
        )
    else:
        return EMAVectorQuantizer(
            codebook_size=config.vq_codebook_size,
            codebook_dim=config.vq_codebook_dim,
            decay=config.vq_decay,
        )


def get_noise_scale(vq_proj_noise, cnt):
    """Get projection noise based on current count."""
    return (vq_proj_noise - cnt).clamp(0) / vq_proj_noise

     
def get_embeddings_from_vector_quantizer(token, vq):
    """Given an integer token, get quantized continuous embeddings after VQ."""
    if isinstance(vq, FiniteScalarQuantizer):
        # this method is badly named. It should be called `indices_to_embeddings` even though FSQ technically outputs codes.
        return vq.indices_to_codes(token)  
    else:
        return vq.embedding(token)


def get_vector_quantizer_projection_layers(vq_proj_norm_type, config):
    """Select how to project into and out of the VQ layer. Otherwise default to Linear Projection."""
    if vq_proj_norm_type == "bn":
        vq_proj_in = nn.Sequential(
            Transpose(),
            WNConv1d(config.hidden_size, config.vq_codebook_dim, kernel_size=1)
            if config.hidden_size != config.vq_codebook_dim
            else nn.Identity(),
            nn.BatchNorm1d(config.vq_codebook_dim, affine=False, momentum=0.05),
            Transpose(),
        )
        vq_proj_out = nn.Sequential(
            Transpose(),
            WNConv1d(config.vq_codebook_dim, config.hidden_size, kernel_size=1)
            if config.vq_codebook_dim != config.hidden_size
            else nn.Identity(),
            Transpose(),
        )
    elif vq_proj_norm_type == "ln":
        vq_proj_in = nn.Sequential(
            nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False)
            if config.hidden_size != config.vq_codebook_dim
            else nn.Identity(),
            nn.LayerNorm(config.vq_codebook_dim, elementwise_affine=False),
        )
        vq_proj_out = nn.Sequential(
            nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False)
            if config.vq_codebook_dim != config.hidden_size
            else nn.Identity()
        )
    else:
        vq_proj_in = nn.Linear(config.hidden_size, config.vq_codebook_dim, bias=False)
        vq_proj_out = nn.Linear(config.vq_codebook_dim, config.hidden_size, bias=False)
    return vq_proj_in, vq_proj_out

def get_vq_codebook_distances(codebook_data):
    """Calculate pairwise codebook distance statistics for monitoring on wandb."""
    embeddings = codebook_data
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    min_distance = torch.min(
        pairwise_distances
        + torch.eye(pairwise_distances.shape[0], device=pairwise_distances.device)
        * pairwise_distances.max()
    )
    return {
        "vq_mean_distance": pairwise_distances.mean(),
        "vq_min_distance": min_distance,
        "vq_max_distance": pairwise_distances.max(),
    }