from typing import Dict

from einops import rearrange
from torch import nn

from samantha.core import BaseStage


class KeyProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        takes=["latent"],
        provides=["key_pred"],
        resample=0,
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_key = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden_channel, 25),
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
        oup["key_pred"] = self.out_key(emb)
        return oup
