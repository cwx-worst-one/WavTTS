import os
from sre_parse import FLAGS

import torch
from absl import app, flags
from cruise.utilities.cloud_io import load as crs_load
from cruise.utilities.hdfs_io import hcopy, hrm

FLAGS = flags.FLAGS  # noqa

flags.DEFINE_string("checkpoint", None, "")


def main(_):
    state_dict = crs_load(FLAGS.checkpoint, map_location="cpu")
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    torch.save(state_dict, "./zero3_merge_states.pt")
    hrm(FLAGS.checkpoint)
    hcopy("zero3_merge_states.pt", os.path.dirname(FLAGS.checkpoint))


if __name__ == "__main__":
    app.run(main)
