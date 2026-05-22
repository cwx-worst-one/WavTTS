from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from wavtts.model import CFM
from wavtts.model.dataset import DynamicBatchSampler, collate_fn
from wavtts.model.utils import default, exists


# trainer


class Trainer:
    @staticmethod
    def _format_params_in_mb(count: int) -> str:
        return f"{count / 1e6:,.3f}M / {count / 1e9:,.6f}B"

    @staticmethod
    def _count_model_params(model: torch.nn.Module) -> tuple[int, int, int]:
        total = 0
        trainable = 0
        frozen = 0
        for param in model.parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
            else:
                frozen += numel
        return total, trainable, frozen

    def _print_model_param_summary(self, model: torch.nn.Module):
        total, trainable, frozen = self._count_model_params(model)
        trainable_ratio = (trainable / total * 100.0) if total > 0 else 0.0
        frozen_ratio = (frozen / total * 100.0) if total > 0 else 0.0

        table_width = 76
        print("\n" + "=" * table_width)
        print("Model Parameter Summary".center(table_width))
        print("-" * table_width)
        print(f"{'Type':<12}{'Count':>18}{'Scale (M / B)':>30}{'Ratio':>16}")
        print("-" * table_width)
        print(
            f"{'Total':<12}{total:>18,}{self._format_params_in_mb(total):>30}{'100.00%':>16}"
        )
        print(
            f"{'Trainable':<12}{trainable:>18,}{self._format_params_in_mb(trainable):>30}{f'{trainable_ratio:.2f}%':>16}"
        )
        print(
            f"{'Frozen':<12}{frozen:>18,}{self._format_params_in_mb(frozen):>30}{f'{frozen_ratio:.2f}%':>16}"
        )
        print("=" * table_width)

    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_wavtts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        model_cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)
            self._print_model_param_summary(self.model)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_wavtts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        if self.is_main:
            print("=" * 80)
            print("Resume training: found checkpoint and loading state")
            print(f"Checkpoint directory: {self.checkpoint_path}")
            print(f"Checkpoint file: {latest_checkpoint}")
            print("=" * 80)

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "WavTTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
            if self.is_main:
                print(f"Resume mode: full-state resume (model + optimizer + scheduler), update={update}")
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0
            if self.is_main:
                print("Resume mode: pretrained-weight init only, update=0")

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from wavtts.infer.utils_infer import cfg_strength, nfe_step, sway_sampling_coef

            target_sample_rate = train_dataset.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    wav = batch["wav"]
                    wav_lengths = batch["wav_lengths"]
                    text_lengths = batch["text_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(wav, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred, loss_dict = self.model(
                        wav, text=text_inputs, lens=wav_lengths, noise_scheduler=self.noise_scheduler,
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update), 
                        aux_mel_loss=loss_dict["aux_mel_loss"].item(),
                        flow_loss=loss_dict["flow_loss"].item(),
                        loss=loss.item()
                    )

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )

                    if loss_dict["flow_loss"] is not None:
                        self.accelerator.log({"flow_loss": loss_dict["flow_loss"].item()}, step=global_update)
                    if loss_dict["aux_mel_loss"] is not None:
                        self.accelerator.log({"aux_mel_loss": loss_dict["aux_mel_loss"].item()}, step=global_update)
                    
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)
                        if loss_dict["flow_loss"] is not None:
                            self.writer.add_scalar("flow_loss", loss_dict["flow_loss"].item(), global_update)
                        if loss_dict["aux_mel_loss"] is not None:
                            self.writer.add_scalar("aux_mel_loss", loss_dict["aux_mel_loss"].item(), global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        unwrap = self.accelerator.unwrap_model(self.model)
                        
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        
                        with torch.inference_mode():
                            ref_wav_len = wav_lengths[0].item()
                            ref_wav = wav[0][:ref_wav_len].unsqueeze(0).to(self.accelerator.device)  # [1, N]

                            generated, _ = unwrap.sample(
                                cond=ref_wav,                   # [1, N]
                                text=infer_text,
                                duration=ref_wav_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32).cpu()  # [1, N_total]

                            cut = ref_wav_len
                            gen_audio = generated[:, cut:]  # [1, N_gen]
                            ref_audio = ref_wav.cpu()       # [1, N_ref]

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
