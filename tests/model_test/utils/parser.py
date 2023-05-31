import logging
import torch
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from samantha.utils.parser import _convert_to_yaml


def parse_precision(dtype):
    precision_map = {'fp32': torch.float32,
                     'fp16': torch.float16
                     }
    try:
        dtype = precision_map[dtype]
    except KeyError:
        logging.error(f'wrong type {dtype}, only support [\'fp32\', \'fp16\']')
        raise
    return dtype


def parse_arguments(arg_list=None):
    r"""Parse command-line arguments for the SAMI AI pytorch-lightning
    experiment.

    Example:
        >>> argv = ['fit', '--config', 'config.yaml', \
                        '--run_opts.fast_dev_run', 'True']
        >>> filename, run_opts, overrides = parse_arguments(argv)
        >>> filename
        'config.yaml'
        >>> run_opts["action"]
        'fit'
        >>> overrides
        {'run_opts': {'fast_dev_run': True}}

    Args:
        arg_list (list, optional): A list of arguments to parse.  If `None`,
            this is read from`sys.argv[1:]`.

    Returns:
        param_file (str): The location of the parameters file.
        run_opts (dict): A dict containing the subcommand (`action`) to run.
        overrides (dict): The overrides to pass to ``load_hyperpyyaml``.
    """
    if arg_list is None:
        arg_list = sys.argv[1:]
    parser = ArgumentParser(
        description="Run a SAMI AI training task using pytorch-lightning.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="A yaml-formatted experiment config file using the "
        "extended YAML syntax defined by SpeechBrain's HyperPyYAML.",
    )

    # # Lightning Trainer's action-specific arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Lightning trainer's argument. If passed, prints more info."
        "Can be used in `validate`, `test`.",
    )

    run_opts, overrides = parser.parse_known_args(arg_list)

    param_file = run_opts.config
    overrides = _convert_to_yaml(overrides)
    return param_file  # , run_opts, overrides
