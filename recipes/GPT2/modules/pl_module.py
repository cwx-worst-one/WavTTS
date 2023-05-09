import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.optim import Adam

from recipes.GPT2.modules.model import GPTLMLoss, get_gpt_model


class GPTLitModule(pl.LightningModule):
    def __init__(
        self,
        model_name,
        betas,
        learning_rate,
        checkpoint,
        provider,
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

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config["zero_optimization"]
            return config.get("offload_optimizer") or config.get("offload_param")
        return False

    def configure_optimizers(self):
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(
                self.model.parameters(), lr=self.learning_rate, betas=self.betas
            )
        elif isinstance(self.trainer.strategy, DeepSpeedStrategy):
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
