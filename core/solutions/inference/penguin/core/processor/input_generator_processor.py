""" penguin input_generator """
# pylint: disable=abstract-method
import os
from core.solutions.inference.penguin.core.processor.processor import Processor
from core.solutions.inference.penguin.core.register import Registers
from core.dataset import build_draw_batch_fn, build_item_augmentation, get_meta
from core.dataset import ValidHDFSDataset


@Registers.processor.register('input_generator')
class InputGenerator(Processor):
    """penguin input_generator"""

    def __init__(self, dolphin_cfg):
        '''init'''
        super().__init__()
        self.dataset_cfg = dolphin_cfg.get('data', None)
        self.draw_batch_fn_inference = build_draw_batch_fn(
            self.dataset_cfg.inference_batch_transform
        )

    def __call__(self, test_dict):
        '''call'''
        return self.process(test_dict)

    def pre_build_dataset(self, dataset_cfg):
        '''prepare for build dataset.'''
        if dataset_cfg.get('meta_file'):
            meta_data_root = dataset_cfg.get("meta_data_root", dataset_cfg.get("data_root", None))
            if isinstance(meta_data_root, (list, tuple)):
                meta_data_root = meta_data_root[0]
            meta_file = os.path.join(meta_data_root, dataset_cfg.meta_file)
            self.meta_data = get_meta(meta_file)
        else:
            self.meta_data = {}
        valid_item_transform = [
            x for x in dataset_cfg.valid_item_transform if x["type"] != "AppendFrames"
        ]
        self.parse_fn_eval = build_item_augmentation(valid_item_transform, self.meta_data)

    def process(self, test_dict):
        '''process'''
        self.pre_build_dataset(self.dataset_cfg)
        self.dataset_cfg["max_batch_size"] = 1
        res = []
        for test_name, test_file_path in test_dict.items():
            test_data_loader = ValidHDFSDataset(
                [test_file_path],
                '',
                self.dataset_cfg,
                self.parse_fn_eval,
                self.draw_batch_fn_inference,
                split_path_list_by_rank=False,
            )
            test_data_loader.reset()
            batch_data = test_data_loader.next()
            while batch_data is not None:
                fbank = batch_data["src"][:, : batch_data["src_mask"].sum().int(), :]
                res.append(('|'.join([batch_data["uttid"][0], test_name]), fbank.tolist()[0]))
                batch_data = test_data_loader.next()
            test_data_loader.terminate()
        return res
