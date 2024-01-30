from pytorch_lightning import Callback, LightningModule, Trainer
from torch.optim import Optimizer


class GradientNormMonitor(Callback):
    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        grad_norm = 0.0
        if pl_module is None:
            return
        for _, p in pl_module.named_parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2)
        pl_module.log_dict(
            {"training/grad_2_norm": grad_norm}, sync_dist=True, prog_bar=True
        )
