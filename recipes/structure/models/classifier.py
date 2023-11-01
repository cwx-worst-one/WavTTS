from typing import Dict

from torch import nn

from samantha.core import BaseStage


class StructureClassifierStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_boundary=2,
        n_function=7,
        takes=["emb"],
        provides=["boundary_pred", "function_pred"],
        serialize_opts=None,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.output_boundary = nn.Sequential(
            nn.LayerNorm(n_channel), nn.Linear(n_channel, n_boundary)
        )
        self.output_function = nn.Sequential(
            nn.LayerNorm(n_channel), nn.Linear(n_channel, n_function)
        )

    def forward(self, data: Dict) -> Dict:
        oup = {}
        emb = data["emb"]
        oup["boundary_pred"] = self.output_boundary(emb)
        oup["function_pred"] = self.output_function(emb)

        return oup
