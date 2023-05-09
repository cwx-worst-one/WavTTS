import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from recipes.byteformers_example.modules.language_modeling import LanguageModelingModule
from recipes.byteformers_example.tokenizer import LlamaTokenizer


class GenerateText(Callback):
    def __init__(self, tokenizer: LlamaTokenizer):
        self.tokenizer = tokenizer

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module: LanguageModelingModule, outputs, batch, batch_idx
    ):

        if trainer.global_step and trainer.global_step % 10 == 0:
            # evaluate both the train and test score
            pl_module.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = self.tokenizer.encode(context, device="cuda")[None, ...]
                y = pl_module.generate(
                    x, 500, temperature=1.0, do_sample=True, top_k=10
                )[0]
                print(self.tokenizer.decode(y.cpu()))
            pl_module.train()
