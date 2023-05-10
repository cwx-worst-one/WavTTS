'''compression module'''
import os.path as osp
import torch
from core.utils import logging, dist_hdfs_get, load_checkpoint, get_rank
from core.dataset.cuda import to_cuda

try:
    from byteslim.core.registry import CONFIG, Torch_Compressor
except ImportError:
    # To use byteslim, include byteslim scm at task building stage.
    # For more detail, see dolphin tutorial.
    # An exception will be raised if byteslim is used.
    pass
CompressorConfigMap = {
    "Quantizer": 'QuantizeConfig',
    'SVDCompressor': 'SVDConfig',
    'Pruner': 'PruneConfig',
}
CompressOrder = ['SVDCompressor', 'Pruner', 'Quantizer']


def is_hdfs_path(path):
    '''check whether the pash is hdfs path'''
    return path.startswith('hdfs')


class DolphinCompressor:
    '''compression class'''

    def __init__(self, solution, cfg):
        '''init.'''
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.slim_config = self.cfg.solution.slim_config
        # For load a pretrain compressed model.
        self.slim_init_config = self.cfg.solution.slim_init_config
        self.solution = solution
        self.init_compressors = {}
        self.compressors = {}

    def init(self):
        '''load resume model before compress'''
        hdfs_file = self.slim_init_config.get('resume_pretrain_ckpt', None)
        if not hdfs_file:
            logging.warning(
                "Before compress, need load pretrain ckpt in resume_pretrain_ckpt, which is None"
            )
        if is_hdfs_path(hdfs_file):
            local_file = 'slim_pretrain_ckpt.pth'
            checkpoint_dir = osp.join(
                self.train_cfg.save_root,
                self.train_cfg.save_dir,
                self.train_cfg.save_name,
                'checkpoints',
            )
            local_path = dist_hdfs_get(hdfs_file, checkpoint_dir, local_file)
        else:
            local_path = hdfs_file

        self.init_compressors = self.compress_model_by_config(self.slim_init_config)

        logging.info("Load checkpoint from {} , before slim".format(hdfs_file))
        load_checkpoint(self.solution, local_path)

        if 'SVDCompressor' in self.init_compressors:
            self.init_compressors['SVDCompressor'].convert_usv_to_uv()

        return local_path

    def compress_model_by_config(self, compressor_config, **kwargs):
        '''create and init all compressors depend on compressor_config'''
        is_training = self.solution.training
        self.solution.eval()

        compressors = {}
        for compressor_name in CompressOrder:
            if compressor_name in compressor_config:
                config = compressor_config[compressor_name]
            else:
                continue
            config_name = CompressorConfigMap[compressor_name]
            layerwise_config = config.pop('layerwise_config', None)
            config = CONFIG.get(config_name)(config, layerwise_config=layerwise_config)
            if compressor_name == 'Pruner':
                handle = self.solution.register_forward_hook(self.trace_hook)
                dummy_input = to_cuda(self.solution.generate_fake_batch_data())
                compressor = Torch_Compressor.get(compressor_name)(
                    self.solution, config, dummy_input
                )
                handle.remove()
            else:
                compressor = Torch_Compressor.get(compressor_name)(self.solution, config)
            if getattr(compressor.config, 'offline', False) and get_rank() != 0:
                offline_dataloader = kwargs.pop('offline_dataloader')
            compressor.compress(**kwargs)
            logging.info(
                "The Compressor: {} compressed the solution, the model compressed:".format(
                    compressor_name
                )
            )
            logging.info(self.solution)
            if getattr(compressor.config, 'offline', False) and get_rank() != 0:
                kwargs['offline_dataloader'] = offline_dataloader
            compressors.update({compressor_name: compressor})
        if is_training:
            self.solution.train()
        return compressors

    def compress(self, **kwargs):
        '''execute compress'''
        self.compressors = self.compress_model_by_config(self.slim_config, **kwargs)

    def post_process(self, optimizer):
        '''process some instance after compress the model'''
        for compressor in self.init_compressors.values():
            compressor.process_optimizer(optimizer)
        for compressor in self.compressors.values():
            compressor.process_optimizer(optimizer)

    @staticmethod
    def trace_hook(_module, _inputs, outputs):
        """hooker for jit trace"""
        new_out = []
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                new_out.append(value)
        return new_out

    def get_slim_loss(self, global_step):
        '''get slim-related regularization loss'''
        slim_loss = 0.0
        for compressor in self.compressors.values():
            _slim_loss = compressor.get_slim_loss(global_step=global_step)
            if _slim_loss is not None:
                slim_loss += _slim_loss
        return slim_loss

    def check_early_stop(self):
        '''check whether to early stop'''
        early_stop = False
        for compressor in self.compressors.values():
            early_stop = early_stop or compressor.check_early_stop()
        return early_stop
