"""Command line argument parser for SAMI AI pytorch-lightning projects."""
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

import yaml


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

    # provides a description for each of the choices in `action`
    actions_helper = {
        "fit": "Runs the full optimization routine (training + validation).",
        "validate": "Perform one evaluation epoch over the validation set.",
        "test": "Perform one evaluation epoch over the test set.",
        "predict": "Run inference on your data (no labels).",
        "export": "Export stage graph and params.",
        "benchmark": "Benchmark tensor throughput of a lightning module",
    }
    parser.add_argument(
        "action",
        choices=actions_helper,
        help="\n".join(
            "{}: {}".format(key, value) for key, value in actions_helper.items()
        ),
    )
    # Lightning Trainer's action-specific arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Lightning trainer's argument. Path to checkpoint to load."
        "Can be used in `fit`, `validate`, `test`, `predict`.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Lightning trainer's argument. If passed, prints more info."
        "Can be used in `validate`, `test`.",
    )
    parser.add_argument(
        "--no_return_predictions",
        dest="return_predictions",
        action="store_false",
        help="Lightning trainer's argument. If passed, not returns predictions."
        "Can be used in `predict`.",
    )
    # SAIL arguments
    parser.add_argument(
        "--post_to_sail",
        action="store_true",
        help="SAIL's argument. If passed, post results to SAIL.",
    )
    parser.add_argument(
        "--arnold_output_dir",
        type=str,
        help="SAIL's argument. Arnold output directory to post results to SAIL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="SAIL's argument. Output directory to post results to SAIL.",
    )

    # Accept extra args to override yaml
    run_opts, overrides = parser.parse_known_args(arg_list)
    param_file = run_opts.config
    overrides = _convert_to_yaml(overrides)
    return param_file, run_opts, overrides


def _expand_branch(tree: dict, branch_seq: list, leaf_value):
    """Expand a key sequence into a dict"""
    branch_depth = len(branch_seq)
    if branch_depth == 0:
        # Should not happen
        raise ValueError(
            f"Got empty `branch_seq` with \
            leaf_value={leaf_value}"
        )
    key = branch_seq[0]
    if branch_depth == 1:
        # Here's the leaf node
        tree[key] = leaf_value
    else:
        sub_tree = tree.get(key, {})
        tree[key] = _expand_branch(sub_tree, branch_seq[1:], leaf_value)
    return tree


def _convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    if not overrides:
        return {}

    yaml_string = ""

    """
    overrides can have 2 formats:
    ['--experiment_id=7171714039348379655', '--ckpt_path=hdfs://haruna/home/byte_arnold_lq/lab/sami/ai_models/tasks/3174942/trials/10708937/output/logs/sample_project/0.1/checkpoints/epoch=1-step=780.ckpt']  # noqa: E501
    or
    ['--experiment_id', '7171714039348379655', '--ckpt_path', 'hdfs://haruna/home/byte_arnold_lq/lab/sami/ai_models/tasks/3174942/trials/10708937/output/logs/sample_project/0.1/checkpoints/epoch=1-step=780.ckpt']  # noqa: E501
    --ckpt_path could be an HDFS path like below:
    hdfs://haruna/home/byte_arnold_lq/lab/sami/ai_models/tasks/3174942/trials/10708937/output/logs/sample_project/0.1/checkpoints/epoch=1-step=780.ckpt  # noqa: E501
    which has '=' in it. We have to do some maneuver
    """
    for arg in overrides:
        if "=" not in arg:
            if arg.startswith("--"):
                yaml_string += "\n" + arg[len("--") :] + ":"
            else:
                yaml_string += " " + arg
        else:
            if arg.startswith("--"):
                # --ckpt_path=hdfs://epoch=1-step=780.ckpt
                first_equal_sign = arg.index("=")
                yaml_string += "\n" + arg[len("--") : first_equal_sign] + ":"
                yaml_string += " " + arg[first_equal_sign + 1 :]
            else:
                yaml_string += " " + arg

    # Handle '--arg.key1.key2=val' type args
    yaml_config = yaml.load(yaml_string.strip(), yaml.Loader)
    config = {}
    for key, value in yaml_config.items():
        config = _expand_branch(config, key.split("."), value)

    return config
