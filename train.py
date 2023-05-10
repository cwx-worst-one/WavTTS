'''
Example Command:
    mpirun --allow-run-as-root -np 8 \
            python3 train.py \
            --config configs/asr_task_hdfs_fusion/las_asr_base.py
'''

import os
import sys
from argparse import ArgumentParser

from core.runner import RUNNERS
from core.utils import Config, distributed_init, get_logger
from core.solutions.inference.penguin.tasks.rnnt.stream_test import run_penguin

try:
    import falconclaw
except ImportError:
    falconclaw = None

base_dir = os.getcwd()
sys.path.append(base_dir)


def _str2bool(bool_str):
    """str to bool"""
    return bool_str.lower() in ("true", "1")


def parse_args():
    '''parse args'''
    parser = ArgumentParser(description='ASR Training scripts')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument(
        '--dump-config',
        type=bool,
        default=False,
        help='dump complete configuration to complete_config.py',
    )
    parser.add_argument('--inference', type=bool, default=False, help='decode mode')
    parser.add_argument('--falcon', type=str, default=str(), help='falcon init token')
    parser.add_argument('--export-onnx', type=bool, default=False, help='export onnx')
    parser.add_argument(
        '--nccl-block', type=_str2bool, default=True, help='nccl block in dist init'
    )
    return parser.parse_known_args()


def main():
    '''main functions'''
    args, unknown = parse_args()
    # init distributed environment if necessary
    port = os.getenv('METIS_WORKER_0_PORT', '0').split(',')[0]
    worker_id = int(os.getenv('DMLC_WORKER_ID', '0'))
    worker_num = int(os.getenv('ARNOLD_WORKER_NUM', '1'))
    gpu_num = int(os.getenv('OMPI_COMM_WORLD_SIZE', '1'))
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0')) + worker_id * gpu_num
    world_size = worker_num * gpu_num
    distributed_init(int(port), block=args.nccl_block, rank=rank, world_size=world_size)

    if rank == 0 or args.inference:
        logger = get_logger(log_level='INFO')
    else:
        logger = get_logger()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_list(unknown)
    if cfg.get('inference', None) and cfg.inference.get('penguin', None):
        args.dump_config = True
        args.export_onnx = True

    if args.dump_config:
        cfg.dump("./complete_config.py")

    if cfg.get('falcon_report', 0) and rank == 0 and falconclaw is not None:
        try:
            falconclaw.init(train_config=cfg.cfg_dict, framework='dolphin', model_type='pytorch')
        except Exception as e:
            logger.error('old falconclaw init error: %r', e)

    logger.info(cfg.filename + ':\n' + cfg.dump())
    logger.info('Enabled distributed training with rank %d world_size %d', rank, world_size)

    runner_cls = RUNNERS.get(cfg.runner)
    if runner_cls is None:
        raise KeyError(f'{cfg.runner} is not in the {RUNNERS.name} registry')
    runner = runner_cls(cfg, inference=args.inference, export_onnx=args.export_onnx)
    logger.info(runner.solution)
    runner.run()

    if cfg.get('inference', None) and cfg.inference.get('penguin', False):
        run_penguin(base_dir, cfg)


if __name__ == '__main__':
    main()
