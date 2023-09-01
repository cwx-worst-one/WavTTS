import os 
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