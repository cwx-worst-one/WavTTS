import functools
import os
import time

import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import AdamW

from recipes.unit2speech.models.diffusion import logger
from recipes.unit2speech.models.ema import EMAModel

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        accelerator,
        model,
        autoencoder,
        diffusion,
        embedder,
        data_loader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_ckpt_dir,
        use_fp16=False,
        end2end=False,
        fp16_scale_growth=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_warmup_steps=0,
    ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.model = model
        self.autoencoder = autoencoder
        self.end2end = end2end
        diffusion.end2end = end2end
        self.ema_model = EMAModel(self.model, ema_rate)
        self.embedder = embedder
        self.data_loader = data_loader
        self.diffusion = diffusion
        self.batch_size = batch_size
        # self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = ema_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_ckpt_dir = resume_ckpt_dir
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=[0.9, 0.95]
        )
        #self.opt = Lion(self.model.parameters(), lr=self.lr)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * self.accelerator.num_processes

        self.sync_cuda = th.cuda.is_available()
        self.accelerator.register_for_checkpointing(
            self.model, self.ema_model, self.opt)
        self._load_parameters()
        self.model = self.accelerator.prepare(self.model)
        if self.embedder is not None:
            self.embedder = self.accelerator.prepare(self.embedder)
        if self.autoencoder is not None:
            self.autoencoder = self.autoencoder.to(self.accelerator.device)
        self.opt = self.accelerator.prepare(self.opt)
        self.ema_model.to(self.accelerator.device)

    def _load_parameters(self):
        if self.resume_ckpt_dir:
            self.resume_step = parse_resume_step_from_dirname(self.resume_ckpt_dir)
            state_dict = th.load(os.path.join(self.resume_ckpt_dir, 'diffusion_model.pt'),
                                 map_location='cpu')
            param_mismatch = False
            model_dict = self.model.state_dict()
            for param_key in set(state_dict.keys()):
                if param_key not in model_dict:
                    del state_dict[param_key]
                    param_mismatch = True
                    self.accelerator.print(f"{param_key} in checkpoint not in model", flush=True)
                else:
                    if state_dict[param_key].shape != model_dict[param_key].shape:
                        state_dict[param_key] = model_dict[param_key]
                        self.accelerator.print(f"{param_key} in checkpoint has different shape", flush=True)
                        param_mismatch = True
            for param_key in model_dict:
                if param_key not in state_dict:
                    state_dict[param_key] = model_dict[param_key]
                    self.accelerator.print(f"{param_key} in model not in checkpoint", flush=True)
                    param_mismatch = True
            self.model.load_state_dict(state_dict)
            if not param_mismatch:
                self.accelerator.print("Resumming from step", self.resume_step, flush=True)
            else:
                self.resume_step = 0
            self.resume_step = 0
        self.accelerator.wait_for_everyone()

    def run_loop(self):
        self.model.train()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            for batch in self.data_loader:
                wavs, tokens = batch
                #print(batch, flush=True)
                self.run_step(wavs, tokens)
                if self.accelerator.is_main_process:
                    if self.step % self.log_interval == 0:
                        logger.dumpkvs()
                    if self.step % self.save_interval == 0:
                        self.save()
                self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, wavs, tokens):
        self.forward_backward(wavs, tokens)
        self._warmup_lr()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, wavs, tokens):
        self.opt.zero_grad()
        xs, x_lens = wavs
        tks, tk_lens = tokens
        xs, tks = xs.to(self.device), tks.to(self.device)
        zs, z_lens = self.autoencoder(xs, x_lens)
        if self.diffusion.compute_loss_count % 10 == 0:
            #self.accelerator.print(f'prompt shape = {prompt.shape}', flush=True)
            self.accelerator.print(f'wav lengths = {x_lens}', flush=True)
            self.accelerator.print(f'latent lengths = {z_lens}', flush=True)
            self.accelerator.print(f'context lengths = {tk_lens}', flush=True)
            self.accelerator.print(
                "Latent Rep.: Shape={} Min={} Max={} Mean={} Std={}".format(
                    zs.shape, zs.min(), zs.max(), zs.mean(), zs.std()), flush=True)
        prompt = []
        for x, x_len in zip(xs, x_lens):
            if x_len > 3 * 24000:
                rand_i = np.random.choice(x_len-3*24000)
                prompt.append(x[rand_i:rand_i+3*24000])
            elif x_len < 3 * 24000:
                prompt.append(th.cat([x[:x_len], th.zeros(3*24000-x_len, device=x.device)], -1))
            else:
                prompt.append(x[:x_len])
        prompt = th.stack(prompt)
        kwargs = {"context": (tks, tk_lens),
                  "prompt": prompt,
                  "z_lens": z_lens}
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            zs.detach(),
            kwargs
        )
        losses, t = compute_losses()
        #print(losses, flush=True)

        loss = losses["loss"].mean()
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        log_loss_dict(
            self.diffusion, t, {k: v for k, v in losses.items()}
        )
        self.accelerator.backward(loss)
        self.opt.step()
        self.model.zero_grad()
        self.ema_model.update(self.model.module)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _warmup_lr(self):
        if not self.lr_warmup_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_warmup_steps
        if frac_done > 1:
            return
        lr = self.lr * frac_done

        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

        logger.log(f"setting lr to {lr}...")

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        seen_samples = (self.step + self.resume_step + 1) * self.global_batch
        logger.logkv("samples", seen_samples)
        if self.step > 0:
            logger.logkv("rank", dist.get_rank())
            logger.logkv("sample rate (per sec)", self.global_batch / (time.time() - self.last_time))
            logger.logkv("pred. time (per 5k)", (time.time() - self.last_time)*5000)
        self.last_time = time.time()

    def save(self):
        ckpt_dir = os.path.join(get_blob_logdir(), f"ckpt{(self.step+self.resume_step):06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.accelerator.save(
            self.accelerator.unwrap_model(self.model).state_dict(), 
            f"{ckpt_dir}/diffusion_model.pt"
        )
        self.accelerator.save(
            self.ema_model.state_dict(),
            f"{ckpt_dir}/ema_diffusion_model.pt"
        )


def parse_resume_step_from_dirname(dirname):
    """
    Parse filenames of the form path/to/ckptNNNNNN, where NNNNNN is the
    checkpoint's number of steps.
    """
    num = dirname.split("/")[-1].replace('ckpt', '')
    try:
        return int(num)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
