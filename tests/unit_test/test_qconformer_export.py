'''
test infer.
more test should add to export onnx.
'''
import os
import copy
import torch

from core.utils import Config
from core.solutions import setup_solution
from core.solutions.inference.infer import INFERS
from core.slim.compressor import DolphinCompressor


def _test_qconformer_exporter():
    '''test jointer infer.'''
    # empty torch cache for Panther BFC GPU mem cache
    torch.cuda.empty_cache()
    # setup solution, and get onnx file
    cfg = Config.fromfile('configs/vad/vad_config_qconformer_export.py')
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('fbank_dim', 640)
    solution_cfg.setdefault('input_concat_size', 1)
    solution_cfg.setdefault('mvn_concat_size', 8)
    solution_cfg.setdefault('model_do_cmvn', True)
    solution_cfg.setdefault('use_symbolic_export', True)

    train_cfg = cfg.train
    onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)
    solution.cuda()
    if solution_cfg.get('slim_config', None):
        compressor = DolphinCompressor(solution, cfg)
        compressor.compress()
    solution.register_infers()

    # get onnx file by export
    solution.cuda()
    solution.export()

    INFERS.clear()


if __name__ == '__main__':
    _test_qconformer_exporter()
