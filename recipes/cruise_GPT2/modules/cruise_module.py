import cruise
from cruise.trainer.strategy import DeepSpeedStrategy
from deepspeed.ops.adam import FusedAdam
from torch.optim import Adam

from recipes.cruise_GPT2.modules.model import GPTLMLoss, get_gpt_model


class GPTLitModule(cruise.CruiseModule):
    def __init__(
        self,
        model_name,
        betas=(0.9, 0.95),
        learning_rate=0.001,
        checkpoint=True,
        provider="huggingface",
        model_type="GPT2LMHead",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = get_gpt_model(
            self.model_name, checkpoint, provider=provider, model_type=model_type
        )
        self.betas = betas
        self.learning_rate = learning_rate
        self.criterion = GPTLMLoss()

    def configure_optimizers(self):
        if isinstance(self.trainer._strategy, DeepSpeedStrategy):
            return FusedAdam(
                self.model.parameters(), lr=self.learning_rate, betas=self.betas
            )
        else:
            return Adam(
                self.model.parameters(), lr=self.learning_rate, betas=self.betas
            )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, input_ids)
        return loss
