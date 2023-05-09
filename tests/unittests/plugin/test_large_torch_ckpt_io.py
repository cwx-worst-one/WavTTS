import os

import numpy as np

from samantha.plugins import LargeTorchCheckpointIO


def test_large_ckpt_io(tmpdir):
    large_bin = {
        "data": np.arange(1 << 30).astype(np.float32)
    }  # (1 << 30) * sizeof(float32) -> 4GiB

    ckpt_io = LargeTorchCheckpointIO()

    out_path = os.path.join(tmpdir, "lbin.ckpt")

    ckpt_io.save_checkpoint(large_bin, out_path)

    loaded_large_bin = ckpt_io.load_checkpoint(out_path)

    assert "data" in loaded_large_bin

    assert (large_bin["data"] == loaded_large_bin["data"]).all()

    ckpt_io.remove_checkpoint(out_path)
    assert not os.path.exists(out_path)
