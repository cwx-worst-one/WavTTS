'''
test streaming rnnt.
'''
import copy
import torch

from core.utils import Config
from core.extensions import AmpEnable
from core.solutions import setup_solution

from core.models.layers.unfold import PantherUnFold


def build_solution():
    '''build solution'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.manual_seed(121738)
    # setup solution, and get onnx file
    config = 'configs/asr/best_rq_repro/debug_bestrq_nonstream_input1k.py'
    cfg = Config.fromfile(config)
    solution_cfg = copy.deepcopy(cfg.solution)
    solution_cfg.setdefault('backend', 'panther')
    solution_cfg.setdefault('fbank_dim', 80)
    solution_cfg['onnx_stack_frame'] = 80
    solution_cfg['onnx_with_mask'] = False
    solution_cfg['backbone_mask'] = False

    solution = setup_solution(solution_cfg)
    # solution.cuda()
    # solution.eval()
    return solution


def test_subsample():
    solution = build_solution()
    with AmpEnable(enabled=False) and torch.no_grad():
        bsz, times, feat_dim = 1, 20, 80
        x = torch.rand([bsz, times, feat_dim])
        x_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]]
        ).float()

        x, x_mask = solution._unfold(x, x_mask)
        x, x_mask = solution._unfold(x, x_mask)
        expected_x_mask = torch.tensor([[1, 1, 1, 0, 1]]).long()
        assert torch.all(x_mask.cpu().long() == expected_x_mask)


if __name__ == '__main__':
    test_subsample()
