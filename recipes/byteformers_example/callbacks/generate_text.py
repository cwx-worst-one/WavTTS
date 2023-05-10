import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from recipes.byteformers_example.modules.language_modeling import LanguageModelingModule
from recipes.byteformers_example.tokenizer import LlamaTokenizer


class GenerateText(Callback):
    def __init__(
        self,
        tokenizer: LlamaTokenizer,
        check_every_n_steps: int,
        num_tokens: int = 500,
        temperature: float = 1.0,
        topk: int = 10,
    ):
        self.tokenizer = tokenizer
        self.check_every_n_steps = check_every_n_steps
        self.num_tokens = num_tokens
        self.temperature = temperature
        self.topk = topk

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module: LanguageModelingModule, outputs, batch, batch_idx
    ):

        if trainer.global_step and trainer.global_step % self.check_every_n_steps == 0:
            pl_module.eval()
            with torch.no_grad():
                context = "O God, O God!"
                x = self.tokenizer.encode(context, device="cuda")[None, ...]
                y = pl_module.generate(
                    x,
                    self.num_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_k=self.topk,
                )[0]
                pl_module.logger.experiment.add_text(
                    "sampled_text",
                    self.tokenizer.decode(y.cpu()),
                    pl_module.global_step,
                )
            pl_module.train()
