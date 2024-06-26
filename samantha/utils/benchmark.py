import pytorch_lightning as pl
from pytorch_lightning.profilers import AdvancedProfiler
from torch.utils.data import DataLoader
from tqdm import tqdm


class BenchmarkDataLoader(pl.LightningDataModule):
    def __init__(self, test_dataset, batch_size: int):
        super().__init__()
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def benchmark_model(
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    pl_datamodule: pl.LightningDataModule,
    output_dir: str,
    n_datapoints: int = 10,
):
    pl_datamodule.setup(stage="fit")
    pl_datamodule.prepare_data()
    # train_loader = pl_datamodule.train_dataloader()

    # test_dataset = []
    # for idx, batch in tqdm(
    #     enumerate(train_loader),
    #     desc="Building benchmark dataset...",
    #     total=n_datapoints,
    # ):
    #     if idx > n_datapoints:
    #         break

    #     test_dataset.append(batch)

    # batch_size = pl_datamodule.batch_size

    # del train_loader
    # del pl_datamodule
    # benchmark_datamodule = BenchmarkDataLoader(test_dataset, batch_size)

    trainer.profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer.fit_loop.epoch_loop.max_epochs = -1
    trainer.fit_loop.epoch_loop.max_steps = 100
    trainer.fit(pl_module, datamodule=pl_datamodule)
