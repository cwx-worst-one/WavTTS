from torch import nn
from typing import Dict
from einops import rearrange
from einops.layers.torch import Rearrange

from samantha.core import BaseStage


class BeatProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        takes=["latent"],
        provides=["beat_pred", "tempo_pred"],
        resample=0,
        serialize_opts=None,
        dropout_rate=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_beat = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden_channel, 3)
        )
        self.out_tempo = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden_channel, 300),
        )
        if resample > 0:
            self.is_resample = True
            self.avg_pool = nn.AdaptiveAvgPool1d(resample)
        else:
            self.is_resample = False

    def forward(self, data: Dict) -> Dict:
        """
        Input:
            data (dict): input dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, channels, length)
        Output:
            oup (dict): output dictionary with two keys "beat_pred" and "tempo_pred".
            oup["beat_pred"] (list): a list includes torch.FloatTensor(batch, length, 3)
            oup["tempo_pred"] (list): a list includes torch.FloatTensor(batch, 300)
        """
        # init dict
        oup = {}

        # linear probing
        emb = data["latent"]
        if self.is_resample:
            emb = self.avg_pool(emb)
        emb = rearrange(emb, "b c t -> b t c")
        oup["beat"] = (self.out_beat(emb), self.out_tempo(emb.mean(1)))
        return {'output': oup}


class BeatConvStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        takes=["latent"],
        provides=["beat_pred", "tempo_pred"],
        resample=0,
        serialize_opts=None,
        dropout_rate=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_beat = nn.Sequential(
            nn.Conv1d(n_channel, n_hidden_channel, 3, 1, 1),
            Rearrange("b c t -> b t c"),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden_channel, 3)
        )
        self.out_tempo = nn.Sequential(
            nn.Conv1d(n_channel, n_hidden_channel, 3, 1, 1),
            Rearrange("b c t -> b t c"),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_hidden_channel, 300),
        )
        if resample > 0:
            self.is_resample = True
            self.avg_pool = nn.AdaptiveAvgPool1d(resample)
        else:
            self.is_resample = False

    def forward(self, data: Dict) -> Dict:
        """
        Input:
            data (dict): input dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, channels, length)
        Output:
            oup (dict): output dictionary with two keys "beat_pred" and "tempo_pred".
            oup["beat_pred"] (list): a list includes torch.FloatTensor(batch, length, 3)
            oup["tempo_pred"] (list): a list includes torch.FloatTensor(batch, 300)
        """
        # init dict
        oup = {}

        # linear probing
        emb = data["latent"]
        if self.is_resample:
            emb = self.avg_pool(emb)
        oup["beat"] = (self.out_beat(emb), self.out_tempo(emb).mean(1))
        return {'output': oup}