import pytorch_lightning as pl

class CurriculumCallback(pl.Callback):
    def __init__(self, datamodule, 
                 epoch_sample_length_increment):
        self.datamodule = datamodule
        self.epoch_sample_length_increment = epoch_sample_length_increment  # sample length increment (in sec) per epoch
        
    def on_train_epoch_start(self, trainer, pl_module):
        # 每5个epoch增加一次样本长度
        if trainer.current_epoch > 0:
            self.datamodule.current_max_length += self.epoch_sample_length_increment
            print(f"### Current sample max length: {self.datamodule.current_max_length}")
