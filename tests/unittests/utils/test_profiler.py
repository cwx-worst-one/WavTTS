import pytorch_lightning as pl
import torch

from samantha.utils.profiler import make_pytorch_profiler


class _Module(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Linear(10, 10)

    def training_step(self, batch):
        batch = torch.tensor(batch, device=self.device)
        loss = self.model(batch).sum()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


def test_make_pytorch_profiler():
    profiler = make_pytorch_profiler(wait=0, warmpup=0, dir_name=None)
    trainer = pl.Trainer(profiler=profiler, max_steps=10)
    trainer.fit(model=_Module(), train_dataloaders=[torch.randn(10) for _ in range(10)])
