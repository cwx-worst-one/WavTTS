from typing import Dict

from recipes.beat.models.networks import Classifier
from samantha.core import BaseStage


class ChordClassifierStage(BaseStage):
    def __init__(
        self,
        n_channel,
        chord_pool,
        takes=["emb"],
        provides=["chord_root", "chord_triad"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_root = Classifier(n_channel, oup_dim=13, pools=chord_pool)
        self.out_triad = Classifier(n_channel, oup_dim=7, pools=chord_pool)
        self.out_boundary = Classifier(n_channel, oup_dim=1, pools=chord_pool)

    def forward(self, data: Dict) -> Dict:
        oup = {}
        emb = data["emb"]

        oup["chord_root"], oup["chord_triad"] = self.out_root(
            data["emb_9"]
        ), self.out_triad(emb)
        return oup


class ChordPerceiverClassifierStage(BaseStage):
    def __init__(
        self,
        n_channel,
        chord_pool,
        takes=["emb"],
        provides=["chord_root", "chord_triad"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_root = Classifier(n_channel, oup_dim=13, pools=chord_pool)
        self.out_triad = Classifier(n_channel, oup_dim=7, pools=chord_pool)
        self.out_boundary = Classifier(n_channel, oup_dim=1, pools=chord_pool)

    def forward(self, data: Dict) -> Dict:
        oup = {}
        emb = data["emb"]

        oup["chord_root"], oup["chord_triad"] = self.out_root(emb), self.out_triad(emb)
        return oup