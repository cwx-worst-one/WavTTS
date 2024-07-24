from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from recipes.research.test_lab.nano_gpt import NanoGPT
from samantha.utils.logger import RankedLogger

logger = RankedLogger(rank_zero_only=True)


class TextGenerationCallback(Callback):
    def __init__(self):
        pass

    @rank_zero_only
    def on_validation_end(self, trainer: Trainer, pl_module: NanoGPT) -> None:
        tokenizer = trainer.datamodule.tokenizer
        text = pl_module.generate(tokenizer)
        logger.info(text)
