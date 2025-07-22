import torch
import pytorch_lightning as pl

def get_task_loss_weights(
        pl_module: pl.LightningModule, 
        prefix: str = "aux/",
    ):
    weight_dict = {}
    for stage in pl_module.model.stages:
        if stage.validattr('loss_weight') and stage.validattr('task'):
            # stage as a model
            weight_dict[f"{prefix}w_loss_{module.task}"] = stage.loss_weight
        if hasattr(stage, 'spans'): 
            # for span_models
            for module in stage.spans:
                if module.validattr('loss_weight') and module.validattr('task'):
                    weight_dict[f"{prefix}w_loss_{module.task}"] = module.loss_weight
        if stage.validattr('insert_modules'):
            # for insert_modules
            for module in stage.insert_modules:
                if module.validattr('loss_weight') and module.validattr('task'):
                    weight_dict[f"{prefix}w_loss_{module.task}"] =  module.loss_weight
    return weight_dict


def get_tasks(pl_module: pl.LightningModule):
    tasks =[]
    for stage in pl_module.model.stages:
        if stage.validattr('task'):
            # stage as a model
            tasks.append(stage.task)
        if hasattr(stage, 'spans'): 
            # for span_models
            tasks += [module.task for module in stage.spans if module.validattr('task')]
        if stage.validattr('insert_modules'):
            # for insert_modules
            tasks += [module.task for module in stage.insert_modules if module.validattr('task')]
    return tasks


def get_task_losses(pl_module: pl.LightningModule, output_dict):
    task_loss_dict = {}
    for stage in pl_module.model.stages:
        if stage.validattr('task') and f"aux/loss_{stage.task}" in output_dict:
            # stage as a model
            task_loss_dict[f"aux/loss_{stage.task}"] = output_dict[f"aux/loss_{stage.task}"]
        if hasattr(stage, 'spans'): 
            # for span_models
            for module in stage.spans:
                if hasattr(module, 'task') and f"aux/loss_{module.task}" in output_dict:
                    task_loss_dict[f"aux/loss_{module.task}"] = output_dict[f"aux/loss_{module.task}"]
        if stage.validattr('insert_modules'):
            # for insert_modules
            for module in stage.insert_modules:
                if hasattr(module, 'task') and f"aux/loss_{module.task}" in output_dict:
                    task_loss_dict[f"aux/loss_{module.task}"] = output_dict[f"aux/loss_{module.task}"]
    return task_loss_dict

def get_quant_rate(
        pl_module: pl.LightningModule, 
        quant_index, 
        quant_token_num
    ):
    """Deprecated in favor of `get_quant_rates`."""
    one_hot = torch.nn.functional.one_hot(
        quant_index.reshape(-1), quant_token_num
    ).sum(dim=0)
    one_hot = pl_module.all_gather(one_hot)
    one_hot = one_hot.sum(dim=0).clamp(0, 1)
    quant_rate = one_hot.sum() / quant_token_num
    return quant_rate


def get_quant_rates(
        pl_module: pl.LightningModule, 
        quant_indices, 
        quant_token_num
    ):
    one_hots = []
    for index in quant_indices:
        one_hot = torch.nn.functional.one_hot(
            index.reshape(-1), quant_token_num
        ).sum(dim=0)
        one_hots.append(one_hot)
    one_hots = pl_module.all_gather(torch.stack(one_hots))
    return one_hots.sum(dim=0).clamp(0, 1).sum() / quant_token_num


def get_quant_rates_robust(
    pl_module: pl.LightningModule,
    quant_indices: torch.Tensor,
    attn_mask: torch.Tensor,
    quant_token_num: int
) -> dict:
    """
    Compute the quantization usage statistics, including quantization rate,
    number of unique tokens used, and number of valid tokens.

    Args:
        pl_module (pl.LightningModule): The Lightning module used for distributed gathering.
        quant_indices (torch.Tensor): Tensor of quantization indices for all tokens.
        attn_mask (torch.Tensor): Attention mask indicating valid token positions.
        quant_token_num (int): Total number of tokens in the quantization codebook.

    Returns:
        dict: A dictionary with:
            - "quant_rate": Ratio of unique tokens used to total available tokens.
            - "total_unique_tokens": Total number of unique tokens used across all devices.
            - "total_valid_tokens": Total number of valid tokens across all devices.
    """
    
    valid_indices = quant_indices[attn_mask]

    num_valid_tokens_local = torch.tensor(
        valid_indices.numel(), dtype=torch.float32, device=pl_module.device
    )

    unique_indices_local = torch.unique(valid_indices)

    one_hot_local = torch.nn.functional.one_hot(
        unique_indices_local, num_classes=quant_token_num
    ).sum(dim=0)  # shape: [quant_token_num]

    one_hots_global = pl_module.all_gather(one_hot_local)  # shape: [world_size, quant_token_num]

    total_unique_tokens_tensor = one_hots_global.sum(dim=0).clamp(0, 1).sum()

    num_valid_tokens_global = pl_module.all_gather(num_valid_tokens_local)
    total_valid_tokens = num_valid_tokens_global.sum().item()

    total_unique_tokens = total_unique_tokens_tensor.item()
    quant_rate = total_unique_tokens / quant_token_num

    return {
        "quant_rate": quant_rate,
        "total_unique_tokens": total_unique_tokens,
        "total_valid_tokens": total_valid_tokens
    }


def get_perplexity(
    pl_module: pl.LightningModule,
    quant_indices: torch.Tensor,
    attn_mask: torch.Tensor,
    quant_token_num: int
) -> float:
    """
    Compute the codebook perplexity on the global batch in a distributed-safe manner.

    Args:
        pl_module (pl.LightningModule): The Lightning module for distributed gathering.
        quant_indices (torch.Tensor): Tensor of quantization indices for all tokens.
        attn_mask (torch.Tensor): Attention mask indicating valid token positions.
        quant_token_num (int): Total number of tokens in the quantization codebook.

    Returns:
        float: The codebook perplexity for the current global batch.
    """
 
    valid_indices = quant_indices[attn_mask]
    if valid_indices.numel() == 0:
        return 0.0

    local_counts = torch.bincount(
        valid_indices, minlength=quant_token_num
    ).float()  # shape: [quant_token_num]

    gathered_counts = pl_module.all_gather(local_counts)
    global_counts = gathered_counts.sum(dim=0)

    total_valid_tokens = global_counts.sum()
    
    if total_valid_tokens == 0:
        return 0.0

    probs = global_counts / total_valid_tokens
    non_zero_probs = probs[probs > 0]
    
    entropy = -torch.sum(non_zero_probs * torch.log2(non_zero_probs + 1e-10))
    
    perplexity = torch.pow(2, entropy).item()
    return perplexity



def get_nuc(target_tokens):
    if target_tokens.dim() == 3:
        nuc = (
            sum(
                [
                    len(target_tokens[i, j, :].unique())
                    for i in range(target_tokens.size(0))
                    for j in range(target_tokens.size(1))
                ]
            )
            / target_tokens.size(0)
            / target_tokens.size(1)
            / target_tokens.size(2)
        )
    elif target_tokens.dim() == 2:
        nuc = (
            sum(
                [
                    len(target_tokens[i, :].unique())
                    for i in range(target_tokens.size(0))
                ]
            )
            / target_tokens.size(0)
            / target_tokens.size(1)
        )
    else:
        raise ValueError(f"Input shape wrong, got {target_tokens.size()}")
    return nuc


def get_codebook_distance_stats_batched(
    codebook_data: torch.Tensor, 
    batch_size: int = 1024
) -> dict:
    """
    Calculate robust pairwise codebook distance statistics in batches to handle large codebooks
    (e.g., > 65k) without causing memory errors.

    Args:
        codebook_data (torch.Tensor): The VQ codebook tensor.
        batch_size (int): The number of vectors to process in each batch.
                          Adjust based on available VRAM.

    Returns:
        dict: A dictionary of codebook distance statistics.
    """
    # 1. Handle zero-magnitude vectors first, as before.
    magnitudes = torch.norm(codebook_data, p=2, dim=1)
    active_mask = magnitudes > 1e-6
    active_embeddings = codebook_data[active_mask]
    num_zero_mag_vectors = codebook_data.shape[0] - active_embeddings.shape[0]
    
    num_active_vectors = active_embeddings.shape[0]

    # 2. Handle edge cases.
    if num_active_vectors < 2:
        return {
            "vq_pairwise_mean_distance": 0.0,
            "vq_pairwise_min_distance": 0.0,
            "vq_pairwise_max_distance": 0.0,
            "num_zero_mag_codebook_vectors": num_zero_mag_vectors,
        }

    # 3. Initialize running statistics.
    global_min_distance = float('inf')
    global_max_distance = 0.0
    global_sum_distance = 0.0

    # 4. Process the codebook in batches to avoid creating a giant N x N matrix.
    for i in range(0, num_active_vectors, batch_size):
        # Select a chunk of the codebook
        chunk = active_embeddings[i : i + batch_size]
        
        # Calculate distance from this chunk to ALL other active vectors
        # This creates a much smaller [batch_size, num_active_vectors] matrix
        dist_chunk = torch.cdist(chunk, active_embeddings, p=2)

        # Update global max
        global_max_distance = max(global_max_distance, dist_chunk.max().item())

        # For min and sum, we must exclude the "diagonal" block
        # that corresponds to the distance of elements within the chunk to themselves.
        # We can do this by setting that block's diagonal to infinity.
        # This is an in-place modification.
        if dist_chunk.shape[0] == dist_chunk.shape[1]: # This happens only when N <= batch_size
             dist_chunk.fill_diagonal_(float('inf'))
        else:
             dist_chunk[:, i : i + batch_size].fill_diagonal_(float('inf'))

        # Update global min
        global_min_distance = min(global_min_distance, dist_chunk.min().item())
        
        # Update global sum. After setting the diagonal to inf, we can't use sum().
        # We reset the diagonal to 0 to calculate the sum of off-diagonal elements correctly.
        if dist_chunk.shape[0] == dist_chunk.shape[1]:
             dist_chunk.fill_diagonal_(0)
        else:
             dist_chunk[:, i : i + batch_size].fill_diagonal_(0)

        global_sum_distance += dist_chunk.sum().item()


    # 5. Calculate the final mean distance.
    num_non_diagonal_elements = num_active_vectors * (num_active_vectors - 1)
    mean_distance = global_sum_distance / num_non_diagonal_elements if num_non_diagonal_elements > 0 else 0.0

    return {
        "vq_pairwise_mean_distance": mean_distance,
        "vq_pairwise_min_distance": global_min_distance,
        "vq_pairwise_max_distance": global_max_distance,
        "num_zero_mag_codebook_vectors": num_zero_mag_vectors,
    }