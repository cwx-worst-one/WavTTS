'''
test infer.
more test should add to export onnx.
'''
import os
import copy
import torch

from core.utils import Config
from core.solutions import setup_solution
from core.solutions.inference.infer import Factory, INFERS


def _test_conformer_exporter(convert_stream=False, with_mask=True):
    '''test jointer infer.'''
    # empty torch cache for Panther BFC GPU mem cache
    torch.cuda.empty_cache()
    # setup solution, and get onnx file
    cfg = Config.fromfile('configs/asr/streaming_conformer_rnnt_input_10k.py')
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    solution_cfg.setdefault('fbank_dim', 80)
    tgt_vocab_size = (
        solution_cfg.adaptive_head_size
        + solution_cfg.adaptive_tail_size * solution_cfg.adaptive_tail_groups
    )
    solution_cfg.setdefault('tgt_vocab_size', tgt_vocab_size)
    solution_cfg.setdefault('conformer_cnn_norm_type', 'layer_norm')
    solution_cfg.setdefault('encoder_convert_stream', convert_stream)
    solution_cfg.setdefault('onnx_with_mask', with_mask)
    solution_cfg.setdefault('export_fused_conformer', True)

    train_cfg = cfg.train
    onnx_dir = os.path.join(train_cfg.save_root, train_cfg.save_dir, train_cfg.save_name, 'onnx')
    solution_cfg.setdefault('onnx_dir', onnx_dir)
    # inference are registered when solution init.
    solution = setup_solution(solution_cfg)
    solution.cuda()
    solution.register_infers()

    # get onnx file by export
    Factory.get_inference('encoder').export()

    INFERS.clear()


def test_conformer_export():
    _test_conformer_exporter(False, True)


def test_conformer_stream_convert():
    _test_conformer_exporter(True, False)


if __name__ == '__main__':
    test_conformer_export()
    test_conformer_stream_convert()
