from typing import Dict

from recipes.beat.models.networks import Classifier
from samantha.core import BaseStage


class BeatClassifierStage(BaseStage):
    def __init__(
        self,
        n_channel,
        beat_pool,
        n_beats=3,
        n_tempo=300,
        takes=["emb"],
        provides=["beat_pred", "tempo_pred"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_beat = Classifier(n_channel, oup_dim=n_beats, pools=beat_pool)
        self.out_tempo = Classifier(n_channel, oup_dim=n_tempo, pools=beat_pool)

    def forward(self, data: Dict) -> Dict:
        oup = {}
        emb = data["emb"]
        oup["beat_pred"], oup["tempo_pred"] = [self.out_beat(emb)], [
            self.out_tempo(emb.mean(1))
        ]
        return oup
