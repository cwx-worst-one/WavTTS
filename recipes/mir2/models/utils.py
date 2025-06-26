from typing import Dict

from sami_ai.core import BaseStage
from recipes.umm2.models.base import PipelineModel


class TransposeStage(BaseStage):
    def __init__(self, takes=["latent"], provides=["latent"], *args, **kwargs):
        super().__init__(takes, provides, *args, **kwargs)

    def forward(self, data: Dict) -> Dict:
        return {"latent": data["latent"].transpose(-2, -1)}


class PipelineModelTupleWrapper(PipelineModel):
    def forward(self, batch):
        output_dict = super().forward(batch)
        return tuple(output_dict[k] for k in self.output_names)


