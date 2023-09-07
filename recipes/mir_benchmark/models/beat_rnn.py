from typing import Dict

from einops import rearrange
from torch import nn

from samantha.core import BaseStage


class BeatRNNStage(BaseStage):
    def __init__(
        self,
        n_channel,
        hidden_channel=1280,
        n_layers=2,
        takes=["latent"],
        provides=["beat_pred", "tempo_pred"],
        resample=0,
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.rnn = nn.LSTM(
            input_size=n_channel,
            hidden_size=hidden_channel,
            num_layers=n_layers,
            bias=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.out_beat = nn.Linear(hidden_channel, 3)
        self.out_tempo = nn.Linear(hidden_channel, 300)
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
        emb = self.rnn(emb)[0]
        oup["beat_pred"], oup["tempo_pred"] = [self.out_beat(emb)], [
            self.out_tempo(emb.mean(1))
        ]
        return oup
