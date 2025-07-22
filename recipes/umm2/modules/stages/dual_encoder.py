from recipes.umm2.modules.pl_module import Stage0
from recipes.umm2.modules.stages.stage3 import get_task_losses, get_task_loss_weights, get_quant_rate, get_nuc
import pytorch_lightning as pl
from pytorch_lightning.profilers import PassThroughProfiler
import torch
import torch.nn.functional as F
from samantha.utils.hparams import DotDict
torch.backends.cudnn.benchmark = True
import numpy as np

def freq_MAE(x, y, sr, fft_size, window, f_max=None):
    freq_dim = fft_size // 2 + 1
    if f_max is not None:
        valid_freq = int(freq_dim * 2 / sr * f_max)
    else:
        valid_freq = freq_dim
    x_spec = torch.stft(x.float(), n_fft=fft_size, hop_length=fft_size//2, window=window.to(x.device), return_complex=True)
    y_spec = torch.stft(y.float(), n_fft=fft_size, hop_length=fft_size//2, window=window.to(y.device), return_complex=True)
    valid_x_spec = x_spec.abs()[:,:valid_freq]
    valid_y_spec = y_spec.abs()[:,:valid_freq]
    norm_scale = valid_y_spec.abs().mean() + 1e-5

    return (valid_x_spec - valid_y_spec).abs().mean() / norm_scale + x_spec.abs().sum() * 0. + y_spec.abs().sum() * 0.

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, sr=24000, fft_size=1024, window="hann_window"):
        """Initialize STFT loss module."""
        super().__init__()

        self.sr = sr
        self.fft_size = fft_size
        self.register_buffer("window", getattr(torch, window)(fft_size))

    def forward(self, x, y, f_max=None):
        # x, y: shape (B, nch, T)
        B, nch, nsample = x.shape
        x = x.reshape(B*nch, nsample)
        y = y.reshape(B*nch, nsample)

        freq_loss = freq_MAE(x, y, self.sr, self.fft_size, self.window, f_max)

        return freq_loss

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        sr=24000, 
        fft_sizes=[32, 64, 128, 256, 512, 1024, 2048],
        window="hann_window",
    ):
        super().__init__()

        self.stft_losses = torch.nn.ModuleList()
        for fft_size in fft_sizes:
            self.stft_losses += [STFTLoss(sr, fft_size, window)]

    def forward(self, x, y, f_max=None):

        freq_loss = 0.
        for f in self.stft_losses:
            freq_loss += f(x, y, f_max) / len(self.stft_losses)

        return freq_loss
    
class DualEncoder(pl.LightningModule):
    def __init__(
        self,
        config,
        model_cls,
        discriminator,
        optimizer_g_cls,
        optimizer_d_cls,
        scheduler_g_cls,
        scheduler_d_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.discriminator = discriminator
        self.config = config
        self.model = model_cls()
        self.required_modules = required_modules

        self.extra_params = DotDict(extra_params)
        self.optimizer_g_cls, self.optimizer_d_cls = optimizer_g_cls, optimizer_d_cls
        self.scheduler_g_cls, self.scheduler_d_cls = scheduler_g_cls, scheduler_d_cls
        
        self.stft_loss = MultiResolutionSTFTLoss(sr=self.model.stages[-1].sample_rate,
                                                 fft_sizes=[64, 128, 256, 512, 1024, 2048, 4096])

        # disable automatic optimization for GAN training
        self.automatic_optimization = False

        self.val_outputs = dict()
        self.flops = 0
        self.ts_before_forward = 0
        self.cached_log_dict = dict()
        if checkpointing:
            self.model.gradient_checkpointing_enable()
        device_name = torch.cuda.get_device_name()
        if "A100" in device_name or "A800" in device_name:
            self.device_FLOPS = 312e12
        elif "H100" in device_name or "H800" in device_name:
            self.device_FLOPS = 989e12
        elif "V100" in device_name:
            self.device_FLOPS = 125e12
        else:
            raise RuntimeError("unknow cuda device name: ", device_name)

    def setup(self, stage: str) -> None:
        if self.global_rank == 0:
            print(self.model)
        if stage == "fit" and self.required_modules is not None:
            self.load_required_modules()

    def load_required_modules(self):
        for module_name, loader_config in self.required_modules.items():
            print(f"loading module {module_name}...")
            _args = {k: v for k, v in loader_config.items() if k != "loader"}
            loader = loader_config["loader"](**_args)
            self = loader.load_model(pl_module=self)
    
    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def configure_optimizers(self):
        # generator
        optimizer_g = self.optimizer_g_cls(self.model.parameters())
        scheduler_g = self.scheduler_g_cls(optimizer_g)
        # discriminator
        optimizer_d = self.optimizer_d_cls(self.discriminator.parameters())
        scheduler_d = self.scheduler_d_cls(optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def training_step_D_and_G(self, batch):
        # get optimizor and scheduler
        net_g, net_d = self.model, self.discriminator
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # train discriminator
        output_dict = net_g(batch)
        sample_rate = self.model.stages[-1].dt_config.sample_rate
        wav_hat = output_dict["audio_recon_"+str(sample_rate)]
        wav = batch["audio_"+str(sample_rate)]
        if wav_hat.ndim == 2:
            wav_hat = wav_hat.unsqueeze(1)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        self.toggle_optimizer(optim_d)
        d_output_real, _ = net_d(wav)
        d_output_fake, _ = net_d(wav_hat.detach())

        D_real = 0.
        D_fake = 0.
        for i in range(len(d_output_real)):
            D_real = D_real + (d_output_real[i].float() - 1).pow(2).mean() / len(d_output_real)
            D_fake = D_fake + (d_output_fake[i].float()).pow(2).mean() / len(d_output_real)
        D_loss = D_real + D_fake

        # disciminator backward
        optim_d.zero_grad()
        self.manual_backward(D_loss)
        self.clip_gradients(optim_d, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optim_d.step()
        scheduler_d.step(self.global_step // 2)
        self.untoggle_optimizer(optim_d)

        # train generator
        self.toggle_optimizer(optim_g)
        d_output_fake, _ = net_d(wav_hat)

        G_fake = 0.
        for i in range(len(d_output_fake)):
            G_fake = G_fake + (d_output_fake[i].float() - 1).pow(2).mean() / len(d_output_fake)

        freq_loss = self.stft_loss(wav_hat, wav)
        G_loss = 3 * freq_loss + G_fake

        
        # for name, param in self.model.named_parameters():
        #     print(name, param.shape)
        #     G_loss = G_loss + param.sum() * 0
        # import pdb; pdb.set_trace()

        # RVQ related losses and metrics
        loss_dict = {
            'freq_loss': freq_loss.item(),
            'G_fake': G_fake.item(),
            'D_loss': D_loss.item(),
        }
        if "loss" in output_dict:
            # code rate & quant rate
            for r in range(output_dict["vq_ids"].shape[-1]):
                quant_rate = get_quant_rate(self,
                    output_dict["vq_ids"][...,r].long(), self.config.vq_codebook_size
                )
                loss_dict[f"aux/quant_rate{r}"] = quant_rate

                if self.trainer.global_step % 100 == 0:
                    code_rate = get_nuc(output_dict["vq_ids"][..., r])
                    loss_dict[f"aux/code_rate{r}"] = code_rate

            for r in range(1, output_dict["vq_ids"].shape[-1]):
                # only 
                if "ppl" in output_dict:
                    loss_dict[f"aux/ppl{r}"] = output_dict["ppl"][..., r-1]
                loss_dict[f"loss_rvq{r}"] = output_dict["loss"][..., r-1]
                if "vq_entropy" in output_dict:
                    loss_dict[f"aux/entropy{r}"] = output_dict["vq_entropy"][..., r-1]

            G_loss = G_loss + output_dict["loss"].sum()

        loss_dict.update(G_loss=G_loss.item())
        # generator backward
        optim_g.zero_grad()
        self.manual_backward(G_loss)
        self.clip_gradients(optim_g, gradient_clip_val=1, gradient_clip_algorithm="norm")
        optim_g.step()
        scheduler_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)

        log_dict = {
                "training/freq_loss": freq_loss,
                "training/D_real": D_real,
                "training/D_fake": D_fake,
                "training/G_fake": G_fake,
                "training/RVQ_loss": output_dict["loss"].sum().item(),
                "aux/step": self.trainer.global_step,
                "aux/max_memory_alloc": torch.cuda.max_memory_allocated() / 2**30,
            }
        log_dict.update(loss_dict)
        # log
        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
        )

    def training_step(self, batch, batch_idx):
        self.training_step_D_and_G(batch)

class Stage4DualEncoder(Stage0):
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        optimizer_d_cls,
        scheduler_d_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            config=config,
            model_cls=model_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.config = config
        self.optimizer_d_cls = optimizer_d_cls
        self.scheduler_d_cls = scheduler_d_cls
        
    def _shared_step(self, batch):
        output_dict = self.model(batch)

        loss_dict = {"loss": output_dict["loss"]}

        mel = output_dict["mel"]
        loss_dict["bs"] = mel.shape[0]
        loss_dict["flops"] = output_dict["flops"]
        
        if "text_ids" in output_dict:
            text_ids = output_dict["text_ids"]
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)

        # RVQ related losses and metrics
        if "loss_rvq" in output_dict and output_dict["loss_rvq"] is not None:
            # code rate & quant rate
            for r in range(output_dict["vq_ids"].shape[-1]):
                quant_rate = get_quant_rate(self,
                    output_dict["vq_ids"][...,r].long(), self.config.vq_codebook_size
                )
                loss_dict[f"aux/quant_rate{r}"] = quant_rate
                if "vq_entropy" in output_dict:
                    loss_dict[f"aux/entropy{r}"] = output_dict["vq_entropy"][..., r]
                if "ppl" in output_dict:
                    loss_dict[f"aux/ppl{r}"] = output_dict["ppl"][..., r]

                loss_dict[f"loss_rvq{r}"] = output_dict["loss_rvq"][..., r]

                if self.trainer.global_step % 100 == 0:
                    code_rate = get_nuc(output_dict["vq_ids"][..., r])
                    loss_dict[f"aux/code_rate{r}"] = code_rate

            output_dict["loss_rvq"] = output_dict["loss_rvq"].sum()
            
        loss_dict.update(get_task_loss_weights(self, prefix="aux/")) # update by adding f"aux/w_loss_{task}"
        loss_dict.update(get_task_losses(self, output_dict))    # update by adding f"loss_{task}"

        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        
        return loss_dict
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch)
        loss_dict = {k: v for k, v in loss_dict.items() if "aux/" not in k} # remove aux items
        
        # skip plotting
        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []
        loss_dict = {k: v for k, v in loss_dict.items() if 'loss' in k}  # keep only loss related items
        self.val_outputs[dataloader_idx].append(loss_dict)

    