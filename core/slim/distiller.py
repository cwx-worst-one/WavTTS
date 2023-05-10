''' distiller module. '''

from copy import deepcopy
from core.utils import load_checkpoint, logging, dist_hdfs_get, Config
from core.solutions import setup_solution
from core.slim.distill_func import distill_loss_func
from core.slim.compressor import DolphinCompressor

try:
    from byteslim.kd.torch.distiller import Distiller
except ImportError:
    # To use byteslim, include byteslim scm at task building stage.
    # For more detail, see dolphin tutorial.
    # An exception will be raised if byteslim is used.
    pass


def get_loss_func(loss_func_args: dict):
    """Used to build Distillation loss function"""
    name = loss_func_args.pop('name')
    return distill_loss_func[name](loss_func_args)


dataset_cfg_attribute = ['fbank_dim', 'use_eos', 'tgt_dict', 'iters_per_epoch']


class DistillerManager:
    """DistillerManager"""

    def __init__(self, student_solution, solution_cfg, local_path, device="cuda"):
        '''init.'''
        self.device = device
        self.student_solution = student_solution
        distiller_config = solution_cfg.get('distiller_config')
        teacher_solution_cfg = distiller_config.get('solution')
        if teacher_solution_cfg is None:
            teacher_solution_cfg = solution_cfg
        for attr in dataset_cfg_attribute:
            if attr in solution_cfg:
                teacher_solution_cfg[attr] = solution_cfg[attr]

        teacher_solution = setup_solution(teacher_solution_cfg)
        if teacher_solution_cfg.get('slim_init_config', None):
            teacher_cfg = dict()
            teacher_cfg['solution'] = teacher_solution_cfg
            teacher_cfg['train'] = {}
            teacher_cfg = Config(teacher_cfg)
            compressor = DolphinCompressor(teacher_solution, teacher_cfg)
            compressor.compress_model_by_config(teacher_solution_cfg.slim_init_config)
        teacher_ckpt_path = distiller_config.get('resume_pretrain_ckpt', None)
        if teacher_ckpt_path is not None:
            local_path = dist_hdfs_get(teacher_ckpt_path, './', 'teacher_ckpt.pth')

        logging.info('Teacher model Load checkpoint from {}'.format(local_path))
        load_checkpoint(teacher_solution, local_path)
        teacher_solution.to(self.device)
        mapping_layers = {}
        loss_func = None
        logits_loss_func = None
        loss_func_ratio = 1.0
        logits_loss_func_ratio = 1.0

        if 'logits' in distiller_config:
            method_config = distiller_config['logits']
            mapping_layers.update({'': ''})  # mapping for logits
            logits_loss_func = get_loss_func(deepcopy(method_config['loss_func']))
            logits_loss_func_ratio = method_config.get('ratio', 1.0)
        if 'feature_map' in distiller_config:
            method_config = distiller_config['feature_map']
            mapping_layers.update(method_config['mapping_layers'])
            loss_func = get_loss_func(deepcopy(method_config['loss_func']))
            loss_func_ratio = method_config.get('ratio', 1.0)

        self.distiller = Distiller(
            self.student_solution,
            teacher_solution,
            mapping_layers,
            loss_func,
            False,
            logits_loss_func,
            loss_func_ratio,
            logits_loss_func_ratio,
        )

        self.student_solution.register_forward_hook(self.hook)

    def caculate_distill_loss(self):
        """
        caculate distill loss
        """
        distill_loss = self.distiller.cal_distill_loss()
        return distill_loss

    def hook(self, _module, inputs, _outputs):
        """hooker for teacher model forward"""
        self.distiller.teacher_model(*inputs)
