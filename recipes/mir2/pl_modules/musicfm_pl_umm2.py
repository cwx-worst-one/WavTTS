import os
from typing import Union, IO
from typing_extensions import Self

import pytorch_lightning as pl
from torch import optim
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from recipes.audio_diffusion.modules.lr_scheduler import TriStageLR

"""There are disk file IO operations in the below import, which takes unpredictable
amount of time. So we time the import here.
"""
import time
start = time.time()
from recipes.mir_benchmark.pl_modules.multitask_finetune_pl import LitFinetuneMultitask
curr = time.time()
dur = curr - start
print(f"Importing LitFinetuneMultitask takes {dur} seconds")


class MusicFMPLUMM2(LitFinetuneMultitask):
    def __init__(self, *args, required_modules=None, **kwargs):
        self.required_modules = required_modules
        super().__init__(*args, **kwargs)

    @classmethod
    def from_default_yaml(cls) -> Self:
        from recipes.mir2.utils.musicfm_adapt import samantha_root_dir
        from recipes.mir2.utils.helper import load_hyperpyyaml_partial
        default_yaml_path = os.path.join(
            samantha_root_dir, "recipes/mir2/conf/musicfm/multi_ft_umm2.yaml"
        )

        # Define a list of keys to skip
        keys_to_keep = [
            "run_opts",
            "train_params",
            "audio",
            "labels",
            "config",
            "frontend",
            "transpose",
            "backend",
            "model",
            "required_modules",
            "pl_module",
        ]
        with open(default_yaml_path) as f:
            hparams = load_hyperpyyaml_partial(f, keys=keys_to_keep)
        assert isinstance(hparams["pl_module"], cls)
        return hparams["pl_module"]

    @classmethod
    def load_from_checkpoint_deepspeed_folder(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
    ) -> Self:
        from recipes.mir2.utils.helper import EmptyDataModule
        pl_module = cls.from_default_yaml()
        trainer = pl.Trainer(
            strategy="deepspeed_stage_2",
        )
        trainer.predict(model=pl_module, datamodule=EmptyDataModule(), ckpt_path=checkpoint_path)
        return pl_module

    def setup(self, stage):
        if self.global_rank == 0:
            print(self.model)
        if stage == "fit" and self.required_modules is not None:
            self.load_required_modules()

    def load_required_modules(self):
        for module_name, loader_config in self.required_modules.items():
            print(f"loading module {module_name}...")
            _args = {k: v for k, v in loader_config.items() if k != "loader"}
            loader = loader_config["loader"](**_args)
            loader.load_model(pl_module=self)

        # def pre_forward_hook(module, input):
        #     print(f"Before forward {module.__class__.__name__}")
            
        #     # Check if input is a tuple (which is typical in PyTorch)
        #     if isinstance(input, tuple):
        #         if len(input) == 1 and isinstance(input[0], dict):
        #             # Handle the case where the input is a single dict
        #             input_dict = input[0]
        #             print("  Input is a dict:")
        #             for key, value in input_dict.items():
        #                 print(f"    Key: {key}, dtype: {value.dtype if isinstance(value, torch.Tensor) else 'Not a Tensor'}")
        #         else:
        #             # Handle non-dict input
        #             print("  Input is a tuple:")
        #             for idx, inp in enumerate(input):
        #                 print(f"    Input {idx} dtype: {inp.dtype if isinstance(inp, torch.Tensor) else 'Not a Tensor'}")
        #     else:
        #         print("  Input is not a tuple. Type:", type(input))

        #     # Check if module has parameters
        #     has_params = False
        #     for name, param in module.named_parameters(recurse=False):
        #         print(f"  Parameter name: {name}, dtype: {param.dtype}")
        #         has_params = True
        #         break  # Print only the first parameter

        #     if not has_params:
        #         print("  No parameters in this module.")

        # for name, module in self.named_modules():
        #     module.register_forward_pre_hook(pre_forward_hook)
    
    def configure_optimizers(self):
        """DANGER: The base class LitFinetuneMultitask hard code stages[1] to be backend.
        Since we added 1 more stage in between, we need to re- hard code it to stages[2].
        """
        params_list = [{"params": self.model.stages[0].parameters(), "lr": self._lr / 10}]
        for task in self.tasks:
            if task == 'beat':
                params_list.append({"params": self.model.stages[2].beat_probing.parameters(), "lr": self._lr / 10})
            if task == 'chord':
                params_list.append({"params": self.model.stages[2].chord_probing.parameters(), "lr": self._lr / 10})
            if task == 'structure':
                params_list.append({"params": self.model.stages[2].structure_probing.parameters(), "lr": self._lr / 10})
            if task == 'vocal2midi':
                params_list.append({"params": self.model.stages[2].vocal2midi_probing.parameters(), "lr": self._lr})
            if task == 'key':
                params_list.append({"params": self.model.stages[2].key_probing.parameters(), "lr": self._lr})

        # Config optimizer and schedulerF
        optimizer = optim.AdamW(params_list, lr=self._lr, weight_decay=0.01, betas=(0.9, 0.98))

        _scheduler = TriStageLR(optimizer, warmup_steps=2000, hold_steps=5000, decay_steps=300000)
        scheduler = {
            "scheduler": _scheduler,
            "interval": "step",
            "name": "learning_rate",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_summary",
        }
