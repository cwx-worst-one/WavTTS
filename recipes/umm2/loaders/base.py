import torch
import os
import contextlib

from abc import ABC, abstractmethod
import torch.distributed as dist
import samantha.utils.hdfs_helper as hh

def is_local_zero():
    local_rank = os.getenv("LOCAL_RANK", None)
    return local_rank is None or local_rank == "0"

@contextlib.contextmanager
def local_zero_first():
    if not dist.is_initialized():
        yield
    else:
        if not is_local_zero():
            dist.barrier()
        yield
        if is_local_zero():
            dist.barrier()

class BaseModelLoader(ABC):
    def __init__(
        self,
        ckpt_path,
        cache_dir,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.cache_dir = cache_dir
        self.device = "cpu"

    @abstractmethod
    def load_model():
        pass

    def ensure_hdfs_ckpt_is_local(self, target_path, cache_dir):
        """If the ckpt path is on HDFS then download it to a local cache, otherwise use the filepath directly."""
        if target_path.startswith("hdfs://"):
            local_path = f"{cache_dir}/{os.path.basename(target_path)}"
            if not os.path.exists(local_path):
                hh.get(target_path, local_path)
                assert os.path.exists(
                    local_path
                ), f"Could not retrieve file from {target_path}."
            return local_path
        else:
            return target_path

    def init_pretrained(self):
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
        with local_zero_first():
            local_path = self.ensure_hdfs_ckpt_is_local(self.ckpt_path, self.cache_dir)
            state_dict = torch.load(local_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            return state_dict

