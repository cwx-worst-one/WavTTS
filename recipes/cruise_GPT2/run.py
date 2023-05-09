from cruise import CruiseCLI, CruiseTrainer
from modules.cruise_datamodule import GPTDataModule
from modules.cruise_module import GPTLitModule

if __name__ == "__main__":
    cli = CruiseCLI(
        GPTLitModule,
        trainer_class=CruiseTrainer,
        datamodule_class=GPTDataModule,
        trainer_defaults={
            "max_epochs": 2,
            "enable_checkpoint": True,
            "checkpoint_monitor": "loss",
            "checkpoint_mode": "min",
            "log_every_n_steps": 1,
            "stats_speedmonitor": True,
            "enable_trace": False,
            "val_check_interval": 0,
        },
    )

    cfg, trainer, model, datamodule = cli.parse_args()
    trainer.fit(model, datamodule)
