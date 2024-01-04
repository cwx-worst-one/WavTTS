import os 
import torch
from pathlib import Path

def download_checkpoint(checkpoint_path, cache_dir):
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    local_path = Path(f'{cache_dir}/{str(Path(checkpoint_path).stem)}.ckpt')
    if not os.path.exists(local_path):
        print(f'Downloading {checkpoint_path}')
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # get the folder path
        folder_path = '/'.join(local_path.parts[:-1])
        if '/home/' in checkpoint_path:
            os.system(f'hdfs dfs -get {checkpoint_path} {folder_path}')
        elif '/mnt/' in checkpoint_path:
            os.system(f'cp {checkpoint_path} {folder_path}')
    return local_path

def random_side_mask(tensor, min_mask_percentage=0.2, max_mask_percentage=0.8):
    """
    Randomly masks the two sides of the last dimension of a tensor for each batch.

    Args:
    - tensor (torch.Tensor): The input tensor with a batch dimension.
    - min_mask_percentage (float): The minimum percentage of the tensor to be masked along the last dimension.
    - max_mask_percentage (float): The maximum percentage of the tensor to be masked along the last dimension.

    Returns:
    - torch.Tensor: The masked tensor.
    """
    
    # Ensure the boundaries are valid
    assert 0 <= min_mask_percentage <= 1, "min_mask_percentage should be between 0 and 1."
    assert 0 <= max_mask_percentage <= 1, "max_mask_percentage should be between 0 and 1."
    assert min_mask_percentage <= max_mask_percentage, "min_mask_percentage should be less than or equal to max_mask_percentage."
    
    # Size of the tensor's last dimension
    length = tensor.shape[-1]
    
    masks = []
    for _ in range(tensor.shape[0]):
        # Select a random mask percentage between min_mask_percentage and max_mask_percentage
        mask_percentage = torch.rand(1).item() * (max_mask_percentage - min_mask_percentage) + min_mask_percentage
        mask_length = int(length * mask_percentage)

        # Randomly decide the number of elements to mask on the left
        left_mask_length = torch.randint(0, mask_length + 1, (1,)).item()
        right_mask_length = mask_length - left_mask_length
        
        mask = torch.ones(length)
        if left_mask_length > 0:
            mask[:left_mask_length] = 0
        if right_mask_length > 0:
            mask[-right_mask_length:] = 0
        masks.append(mask)

    mask_tensor = torch.stack(masks).unsqueeze(1)
    # Apply the mask
    masked_tensor = tensor * mask_tensor.to(tensor.device)

    return masked_tensor
