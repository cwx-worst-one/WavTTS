# This file computes per-song FAD and similarity with pre-extracted embeddings.
# It also performs embedding post-processing, 
#   such as appending time encoding and augmenting with pooled embeddings.
# @author: Yatong Bai

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from torch import Tensor


def get_embedding_similarity(
    audio_embd: List[Tensor],  # b * [n_i, d]
    text_embd: Tensor,  # [b, d]
) -> Dict[str, Tensor]:
    """
    Compute similarity score between audio embeddings and text embeddings.
    If either audio or text embeddings contains NaN, the similarity score will be NaN.

    Args:
        audio_embd (List[Tensor]):
            List of audio embedding tensors with length b, each with shape [n_i, d].
        text_embd (Tensor):
            Tensor of text embeddings with shape [b, d].

    Returns:
        Dict: A dictionary of Tensors containing similarity scores.
    """
    assert isinstance(audio_embd, list), f"Expected list, but got {type(audio_embd)}"
    assert all(isinstance(a, Tensor) for a in audio_embd), f"Expected list of Tensors"
    assert isinstance(text_embd, Tensor), f"Expected Tensor, but got {type(text_embd)}"
    assert len(audio_embd) == text_embd.shape[0], f"{len(audio_embd)} vs {text_embd.shape[0]}"

    # Compute similarity score using raw audio embeddings and then average per piece
    sim_no_pool = [F.cosine_similarity(a, t) for a, t in zip(audio_embd, text_embd)]
    sim_no_pool_mean = torch.stack([sim.mean() for sim in sim_no_pool])
    sim_no_pool_median = torch.stack([sim.median() for sim in sim_no_pool])

    # Compute similarity score using pooled audio embeddings
    audio_embd_pooled = torch.cat(
        [a.mean(dim=0, keepdim=True) for a in audio_embd], dim=0
    )
    sim_pool = F.cosine_similarity(audio_embd_pooled, text_embd.squeeze(1))

    return {
        "Raw": sim_no_pool,
        "Raw_Mean": sim_no_pool_mean,
        "Raw_Median": sim_no_pool_median,
        "Pooled": sim_pool
    }


def hermitian_matrix_sqrt(matrix: Tensor) -> Tensor:
    """
    Compute the square root of a Hermitian matrix.

    Args:
        matrix (Tensor):
            A Hermitian matrix to compute the square root of. Shape: (dim, dim)

    Returns:
        Tensor: The square root of the input matrix.
    """
    assert matrix.dim() == 2, f"Matrix must be 2D, but got {matrix.dim()}D."
    assert matrix.shape[0] == matrix.shape[1], f"Matrix must be square, but got {matrix.shape}"
    assert torch.allclose(matrix, matrix.T, atol=1e-6), "Matrix must be symmetric."

    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals_sqrt = torch.sqrt(torch.clamp(eigvals, min=0))  # Clamp to avoid tiny negative values
    matrix_sqrt = eigvecs @ torch.diag(eigvals_sqrt) @ eigvecs.T
    return matrix_sqrt


def attach_time_encoding(embd_tensor: Tensor, incre: float = 3e-3) -> Tensor:
    """
    Attach a linear time encoding to the input tensor.

    Args:
        embd_tensor (Tensor):
            Input tensor with shape (seq_len, embedding_dim).
        incre (float, optional):
            Increment value for time embedding. Defaults to 3e-3.

    Returns:
        Tensor: Tensor with shape (seq_len, embedding_dim + 1).
    """
    time_embedding = torch.arange(embd_tensor.shape[0]).to(embd_tensor.device) * incre
    time_embedding = time_embedding.unsqueeze(-1) - time_embedding.mean()
    embd_tensor = torch.cat([embd_tensor, time_embedding], dim=1)
    return embd_tensor


def extend_with_moving_avg_pool(embd: Tensor, window_len: int = 5) -> Tensor:
    """
    Performs overlapping average pooling on a tensor and concatenates it with the original tensor.

    This function takes a 2D tensor `embd` of shape (T, C), where T is the sequence
    length and C is the number of channels (e.g., 512). It computes an average pooling
    over a sliding window of size window_len with a stride of 1.

    The pooled tensor, of shape (T-window_len+1, C), is then concatenated with the original
    tensor (with its last window_len-1 frames dropped) along the channel dimension.

    Args:
        embd (torch.Tensor):
            The input tensor of shape (T, C) where C=512 for MuLan.
        window_len (int, optional):
            The window length for pooling. Defaults to 5.

    Returns:
        torch.Tensor:
            The resulting tensor of shape (T-window_len+1, 2*C).
            I.e., (T-4, 1024) with default setting for MuLan.
    """
    # If sequence is shorter than window_len, pool all frames and concatenate to the first frame.
    if embd.shape[0] < window_len:
        pooled_embd = embd.mean(dim=0)
        cat_embd = torch.cat([embd[0], pooled_embd]).unsqueeze(0)
        return cat_embd

    # Use unfold to create a view of the tensor with sliding windows. Apply to dim=0 (sequence dim)
    # with window size window_len and step 1. Resulting shape (T-window_len+1, C, window_len).
    unfolded_embd = embd.unfold(dimension=0, size=window_len, step=1)

    # Compute the mean along the last dimension (window). This averages the window_len
    # consecutive tensors in each window. Resulting shape (T-window_len+1, C).
    pooled_embd = unfolded_embd.mean(dim=2)

    # Concatenate the original tensor (drop last window_len-1 frames) and the pooled
    # tensor along dim=1. (T-4, C) + (T-4, C) -> (T-4, 2*C).
    cat_embd = torch.cat((embd[:-window_len+1], pooled_embd), dim=1)
    return cat_embd


def fast_frechet_distance(
    mu1: Tensor,  # shape (*, embedding_dim)
    sigma1: Tensor,  # shape (*, embedding_dim, embedding_dim)
    mu2: Tensor,  # shape (embedding_dim,)
    sigma2: Tensor,  # shape (embedding_dim, embedding_dim)
    sigma2_sqrt: Optional[Tensor] = None,  # shape (embedding_dim, embedding_dim)
    match_bottom_right: bool = False  # For time-augmented FAD
) -> Tensor:
    """
    Calculates the Fréchet Audio Distance (FAD) between two multivariate Gaussian distributions:
    FAD^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    This function supports batch operations for the first distribution (mu1, sigma1).

    Args:
        mu1 (Tensor):
            The mean of the distribution to evaluate. Shape: (*, embedding_dim)
        sigma1 (Tensor):
            The covariance matrix of the distribution to evaluate.
            Shape: (*, embedding_dim, embedding_dim)
        mu2 (Tensor):
            The mean of the reference distribution. Not batched.
            Shape: (embedding_dim,)
        sigma2 (Tensor):
            The covariance matrix of the reference distribution. Not batched.
            Shape: (embedding_dim, embedding_dim)
        sigma2_sqrt (Optional[Tensor]): 
            An optional pre-computed matrix square root of sigma2.
            If provided, it avoids re-computation. Defaults to None.
        match_bottom_right (bool, optional):
            Whether to match the bottom right element of the covariance matrices.
            This is useful for embeddings with appendix time encoding. Defaults to False.

    Returns:
        Tensor: A scalar or 1D tensor containing the Fréchet Audio Distance for each item in the batch.
    """

    # Check dimensions
    assert sigma1.dim() - mu1.dim() == 1, f"got {sigma1.dim()}D sigma1 but {mu1.dim()}D mu1"
    assert mu2.dim() == 1, f"mu2 must be 1D, got {mu2.dim()}D"
    assert sigma2.dim() == 2, f"sigma2 must be 2D, got {sigma2.dim()}D"
    assert sigma2.shape[0] == sigma2.shape[1], f"sigma2 must be square, got {sigma2.shape}"
    assert mu1.shape == sigma1.shape[:-1], \
        f"mu1 and sigma1 must have the same shape until the last dimension, got {mu1.shape} and {sigma1.shape}"
    assert mu1.shape[-1] == sigma1.shape[-1], \
        f"sigma1 and mu1 must have the same last dimension, got {sigma1.shape[-1]} and {mu1.shape[-1]}"
    assert mu1.shape[-1] == mu2.shape[-1], \
        f"mu1 and mu2 must have the same last dimension, got {mu1.shape[-1]} and {mu2.shape[-1]}"
    assert sigma2.shape[-1] == mu2.shape[-1], \
        f"sigma2 and mu2 must have the same last dimension, got {sigma2.shape[-1]} and {mu2.shape[-1]}"

    if match_bottom_right:
        sigma2_sqrt = None
    if sigma2_sqrt is not None:
        assert sigma2_sqrt.shape == sigma2.shape, \
            f"sigma2_sqrt must have the same shape as sigma2, got {sigma2_sqrt.shape} and {sigma2.shape}"

    # Ensure tensors are on the same device
    if mu1.device != sigma1.device or mu1.device != mu2.device or mu1.device != sigma2.device:
        raise ValueError("All input tensors must be on the same device.")

    # Convert everything to float64 for numerical stability
    original_dtype = mu1.dtype
    mu1 = mu1.to(torch.float64)
    sigma1 = sigma1.to(torch.float64)
    mu2 = mu2.to(torch.float64)
    sigma2 = sigma2.to(torch.float64)
    if sigma2_sqrt is not None:
        sigma2_sqrt = sigma2_sqrt.to(torch.float64)

    # Assert no NaN values
    assert not torch.isnan(mu1).any(), "mu1 contains NaN values"
    assert not torch.isnan(sigma1).any(), "sigma1 contains NaN values"
    assert not torch.isnan(mu2).any(), "mu2 contains NaN values"
    assert not torch.isnan(sigma2).any(), "sigma2 contains NaN values"
    if sigma2_sqrt is not None:
        assert not torch.isnan(sigma2_sqrt).any(), "sigma2_sqrt contains NaN values"

    # Enforce symmetry for covariance matrices
    if not torch.allclose(sigma1, sigma1.transpose(-2, -1), rtol=1e-5, atol=1e-8):
        actual_diff = torch.abs(sigma1 - sigma1.transpose(-2, -1))
        raise ValueError(f"sigma1 is not symmetric. max diff={actual_diff.amax(dim=[-1, -2])}")
    if not torch.allclose(sigma2, sigma2.T, rtol=1e-5, atol=1e-8):
        actual_diff = torch.abs(sigma2 - sigma2.T)
        raise ValueError(f"sigma2 is not symmetric. max diff={actual_diff.amax()}")
    sigma1 = (sigma1 + sigma1.transpose(-2, -1)) / 2
    sigma2 = (sigma2 + sigma2.T) / 2

    if sigma2_sqrt is not None:
        if not torch.allclose(sigma2_sqrt, sigma2_sqrt.T):
            actual_diff = torch.abs(sigma2_sqrt - sigma2_sqrt.T)
            raise ValueError(f"sigma2_sqrt is not symmetric. max diff={actual_diff.amax()}")
        sigma2_sqrt = (sigma2_sqrt + sigma2_sqrt.T) / 2

    # For time-augmented FAD, match the bottom right element of the covariance matrices.
    # This avoids the influence of the audio length difference between the two distributions.
    if match_bottom_right:
        mu1[..., -1], mu2[-1] = 0, 0
        sig1_br, sig2_br = sigma1[..., -1, -1], sigma2[-1, -1]
        max_sig_br = torch.max(sig1_br.max(), sig2_br)
        sigma1[..., -1, -1] = max_sig_br
        sigma2[-1, -1] = max_sig_br

    # 1. Calculate the squared Mahalanobis distance between the means.
    # Broadcasting mu2 to match the batch dimension of mu1.
    dist_mu_sq = torch.sum((mu1 - mu2) ** 2, dim=-1)

    # 2. Calculate covariance traces. Take diagonal and sum as torch.trace is not batched.
    trace_sigma1 = torch.diagonal(sigma1, dim1=-2, dim2=-1).sum(-1)
    trace_sigma2 = torch.trace(sigma2)  # Not batched

    # 3. Calculate Tr(sqrt(sigma1 * sigma2))
    if sigma2_sqrt is None:
        sigma2_sqrt = hermitian_matrix_sqrt(sigma2)

    # Batched matrix mul: (B, D, D) @ (D, D) -> (B, D, D); eigvalsh is already batched.
    eigvals_m = torch.linalg.eigvalsh(sigma2_sqrt @ sigma1 @ sigma2_sqrt)

    # Sum of sqrt of eigenvalues for the trace.
    trace_sqrt_cov_prod = torch.sum(torch.sqrt(torch.clamp(eigvals_m, min=0)), dim=-1)

    # 4. Combine the parts to get the final FAD score.
    fad_sq = dist_mu_sq + trace_sigma1 + trace_sigma2 - 2 * trace_sqrt_cov_prod

    return torch.clamp(fad_sq, min=0).to(original_dtype)


def get_fad_from_embeddings(
    audio_embd: List[Tensor],  # b * [n_i, d]
    ref_stats: Dict,
    match_bottom_right: bool = False,
) -> Dict[str, Tensor]:
    """
    Compute the Fréchet Audio Distance (FAD) between audio embeddings and reference statistics.
    If the audio embedding contains NaN or is too short (less than 2 frames), the FAD will be NaN.

    Args:
        audio_embd (List[Tensor]):
            List of audio embedding tensors with length b, each with shape [n_i, d].
        ref_stats (Dict):
            A dictionary of reference statistics, where each key is a reference name,
            and the value is a dictionary with keys 'mean', 'cov', and 'cov_sqrt'.
        match_bottom_right (bool, optional):
            Whether to match the bottom right element of the covariance matrices.
            This is useful for embeddings with appendix time encoding. Defaults to False.

    Returns:
        Dict: A dictionary of Tensors containing FAD scores for each reference.
    """
    # Evict embeddings with NaNs from long_sampled_embeds.
    idx_long_audio = [i for i, em in enumerate(audio_embd) if len(em) > 1 and not torch.isnan(em).any()]
    long_audio_embd = [audio_embd[i] for i in idx_long_audio]

    long_audio_means = torch.stack([a.mean(dim=0) for a in long_audio_embd], dim=0)
    long_audio_covs = torch.stack([a.T.cov() for a in long_audio_embd], dim=0)

    psfad_dict = {}
    for key, stats in ref_stats.items():
        long_audio_psfad = fast_frechet_distance(
            mu1=long_audio_means, sigma1=long_audio_covs,
            mu2=stats['mean'], sigma2=stats['cov'], sigma2_sqrt=stats['cov_sqrt'], 
            match_bottom_right=match_bottom_right,
        )
        # Fill NaN for audio embeddings that contain NaNs or are too short
        all_psfad = torch.zeros(len(audio_embd), device=long_audio_psfad.device).fill_(torch.nan)
        all_psfad[idx_long_audio] = long_audio_psfad
        psfad_dict[key] = all_psfad

    return psfad_dict
