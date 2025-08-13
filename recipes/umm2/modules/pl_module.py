import time

import pytorch_lightning as pl
import torch
import torch.distributed
from pytorch_lightning.profilers import PassThroughProfiler

from samantha.utils.hparams import DotDict
from recipes.umm2.modules.lr_scheduler import get_model_parameters_with_lr
from recipes.umm2.modules.logging_utils import (
    get_wandb_logger,
    get_list_of_mel_spec_plots_to_log,
    get_list_of_chroma_spec_plots_to_log,
)

import samantha
from mariana.utils.comm_utils import get_formated_model_summary_table
from mariana.utils.audio.audio_logger import AudioLogger

logger = AudioLogger()

class Stage0(pl.LightningModule):
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = model_cls()
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.extra_params = DotDict(extra_params)
        self.required_modules = required_modules
        self.val_outputs = dict()
        self.flops = 0
        self.ts_before_forward = 0
        self.cached_log_dict = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()
        device_name = torch.cuda.get_device_name()
        if "A100" in device_name or "A800" in device_name or "H20" in device_name:
            self.device_FLOPS = 312e12
        elif "H100" in device_name or "H800" in device_name:
            self.device_FLOPS = 989e12
        elif "V100" in device_name:
            self.device_FLOPS = 125e12
        elif "H20" in device_name and "H200" not in device_name:
            self.device_FLOPS = 148e12
        else:
            raise RuntimeError("unknow cuda device name: ", device_name)


    def setup(self, stage: str) -> None:
        if self.global_rank == 0:
            logger.info(self.model)
        if stage == "fit" and self.required_modules is not None:
            self.load_required_modules()
            summary = get_formated_model_summary_table(self.model)
            logger.info(summary)


    def load_required_modules(self):
        self.requires = {}
        for module_name, loader_config in self.required_modules.items():
            print(f"loading module {module_name}...")
            if "loader" in loader_config:   # load into model
                _args = {k: v for k, v in loader_config.items() if k != "loader"}
                loader = loader_config["loader"](**_args)
                self = loader.load_model(pl_module=self)
            else:   # load as extra module
                hpath = loader_config['hpath']
                initializer = loader_config['initializer']
                self.requires.update(initializer(hpath, local_rank=self.local_rank, cache_dir="./"))


    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        # Not quite accurate, time used by optimizer is also counted.
        elapsed = time.time() - self.ts_before_forward
        mfu = self.flops / elapsed / self.device_FLOPS
        self.log_dict_cached(
            {
                "training/mfu": mfu,
                # FIXME: The below two should really be "max" or "use rank 0".
                "training/mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "training/malloc_retries": torch.cuda.memory_stats()[
                    "num_alloc_retries"
                ],
            }
        )

        # This makes sure we don't lose any log items added after forward pass.
        #
        # As a bonus, since by the time we reach here, the previous pass is
        # guaranteed to complete, so we won't need to wait on CUDA computation
        # synchronously.
        #
        # We defer actual logging operation to overlap it with CUDA computation.
        prev_log_dict, pending_deletion = self.flush_log_dict()

        self.ts_before_forward = time.time()
        loss_dict = self._shared_step(batch)

        # Overlap these operations with CUDA computation.
        for i in batch.keys():
            batch[i] = None
        self.log_dict(
            prev_log_dict, prog_bar=True, sync_dist=False, rank_zero_only=True
        )
        del pending_deletion
        del prev_log_dict

        if "flops" in loss_dict:
            self.flops = loss_dict["flops"]
            del loss_dict["flops"]

        loss_dict["training/loss"] = loss_dict["loss"]
        self.log_dict_cached(loss_dict)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch)
        loss_dict = {k: v for k, v in loss_dict.items() if "aux/" not in k} # remove aux items
        
        if batch_idx == 0:           
            # Get the instance of the wandb_logger
            wandb_logger = get_wandb_logger(self.logger)
            num_samples_to_plot = self.config.get('num_spectrogram_val_samples_for_plotting', 8)

            # Log Mel Spectrograms
            if "mel" in loss_dict.keys():
                gt_mel_wandb_img_list = get_list_of_mel_spec_plots_to_log(loss_dict['mel'], num_samples_to_plot)
                recon_mel_wandb_img_list = get_list_of_mel_spec_plots_to_log(loss_dict['mel_out'], num_samples_to_plot)
                wandb_logger.experiment.log({"Mel GT": gt_mel_wandb_img_list})
                wandb_logger.experiment.log({"Mel Recon": recon_mel_wandb_img_list})

            # Log Chroma 
            if "chroma" in loss_dict.keys():
                gt_chroma_wandb_img_list = get_list_of_chroma_spec_plots_to_log(loss_dict['chroma'], num_samples_to_plot)
                recon_chroma_wandb_img_list = get_list_of_chroma_spec_plots_to_log(loss_dict['chroma_out'], num_samples_to_plot)
                wandb_logger.experiment.log({"Chroma GT": gt_chroma_wandb_img_list})
                wandb_logger.experiment.log({"Chroma Recon": recon_chroma_wandb_img_list})

        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        loss_dict = {k: v for k, v in loss_dict.items() if 'loss' in k}  # keep only loss related items
        self.val_outputs[dataloader_idx].append(loss_dict)

    def on_validation_epoch_end(self):
        for dataloader_idx, outputs in self.val_outputs.items():
            val_loss_dict = {}
            for loss in outputs:
                for k, v in loss.items():
                    k = f"val_{dataloader_idx}/{k}"
                    if k not in val_loss_dict:
                        val_loss_dict[k] = v
                    else:
                        val_loss_dict[k] = val_loss_dict[k] + v
            for k, v in val_loss_dict.items():
                val_loss_dict[k] = v / len(outputs)
            self.log_dict(val_loss_dict, prog_bar=True, sync_dist=True)
            self.val_outputs[dataloader_idx] = []

    def configure_optimizers(self):
        # each basestage module has a lr_ratio, and this process determine the lr_ratio for the corresponding module parameters
        # true_lr = lr * lr_ratio, enabling different learning rates for different modules
        params, lrs = get_model_parameters_with_lr(self) 
        optimizer = self.optimizer_cls(params)
        scheduler = self.scheduler_cls(optimizer=optimizer, init_lr=lrs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        } 


    def log_dict_cached(self, kvs):
        """Save `kvs` into cached log dict. The dict is flushed after each
        forward pass."""
        def is_scalar(v):
            if isinstance(v, torch.Tensor):
                return v.numel() == 1
            # Add other potential scalar types if necessary
            return isinstance(v, (int, float))

        self.cached_log_dict.update(
            {k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in kvs.items() if is_scalar(v)}
        )
    def flush_log_dict(self):
        orig_keys = []
        cpu_values = []
        cuda_values = []

        # Move all non-CUDA tensor in one go.
        for k, v in self.cached_log_dict.items():
            if v is not torch.Tensor:
                orig_keys.append(k)
                cpu_values.append(v)
            elif not v.is_cuda:
                orig_keys.append(k)
                cpu_values.append(v.item())
            else:
                assert v.is_cuda

        for k, v in self.cached_log_dict.items():
            if v is torch.Tensor and v.is_cuda:
                orig_keys.append(k)
                if len(v.size()) == 0:
                    v = torch.unsqueeze(v.float(), 0)
                cuda_values.append(v)

        values = torch.cat(
            [torch.tensor(cpu_values, dtype=torch.float, device="cuda")] + cuda_values
        )
        # Support different collectives is just a matter of gathering metrics
        # to rank 0 and reducing them locally.
        torch.distributed.reduce(values, 0, op=torch.distributed.ReduceOp.AVG)

        res_dict = {orig_keys[i]: values[i] for i in range(len(values))}
        self.cached_log_dict.clear()

        return res_dict, [orig_keys, cpu_values, cuda_values, values]
