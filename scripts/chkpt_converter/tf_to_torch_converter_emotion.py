"""
This script can be used to convert a head-less TF1.x model to PyTorch 

especially to convert bert model from bytebot-nlu

The script re-maps the TF weight names to the original names,
so the model can be imported with dolphin.

NOTE: Dolphin solution must include the function tf_chkpt_converter,
which converts tf variables dictionary to torch state dictionary.
"""


import argparse
import os
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), ".."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../.."))

import tensorflow as tf
import torch

from core.utils import logging, Config, hdfs_get, get_logger
from core.solutions import setup_solution

logger = get_logger(log_level='INFO')


def load_tf1_weights_in_dolphin(model, tf_checkpoint_path):
    """Load tf1.x checkpoints in a pytorch model."""
    if tf_checkpoint_path.startswith("hdfs://"):
        hdfs_get(tf_checkpoint_path)
        tf_checkpoint_path = os.path.basename(tf_checkpoint_path).replace('*', '')

    tf_path = os.path.abspath(tf_checkpoint_path)
    logging.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    tf_var_dict = {}
    for name, shape in init_vars:
        logging.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_var_dict[name] = array

    # Function tf_chkpt_converter(tf_var_dict):
    # Args:
    #   tf_var_dict(dictionary): a dictionary containing all tf checkpoint variables.
    # Returns:
    #   dolphin_state_dict(state_dict): dolphin state dict class.

    model.tf_chkpt_converter(tf_var_dict)
    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, dolphin_config_path, pytorch_dump_path):
    '''convert_tf_checkpoint_to_pytorch'''
    # Instantiate model
    logging.info(f"Loading model based on config from {dolphin_config_path}...")
    cfg = Config.fromfile(dolphin_config_path)
    solution_cfg = cfg.solution
    model = setup_solution(solution_cfg)

    if not hasattr(model, 'tf_chkpt_converter'):
        raise RuntimeError(
            "The specified dolphin solution does not have "
            "a tensorflow checkpoint convert function"
        )

    # Load weights from checkpoint
    logging.info(f"Loading weights from tf1.x checkpoint {tf_checkpoint_path}...")
    load_tf1_weights_in_dolphin(model, tf_checkpoint_path)

    # Save pytorch-model
    logging.info(f"Saving PyTorch model to {pytorch_dump_path}...")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path",
        type=str,
        required=True,
        help="Path to the TensorFlow 1.x checkpoint path.",
    )
    parser.add_argument(
        "--dolphin_config_file",
        type=str,
        required=True,
        help="The config file corresponding to the dolphin model. "
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.dolphin_config_file, args.pytorch_dump_path
    )
