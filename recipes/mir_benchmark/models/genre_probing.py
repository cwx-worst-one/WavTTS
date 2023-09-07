from typing import Dict

from einops import rearrange
from torch import nn

from samantha.core import BaseStage


class GenreProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        n_out=10,
        takes=["latent"],
        provides=["genre_pred"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_genre = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden_channel, n_out),
        )

    def forward(self, data: Dict) -> Dict:
        """
        Input:
            data (dict): input dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, channels, length)
        Output:
            oup (dict): output dictionary with a key "genre_pred".
            oup["genre_pred"] (list): a list includes torch.FloatTensor(batch, class)
        """
        # init dict
        oup = {}

        # linear probing
        emb = data["latent"].mean(-1)
        oup["genre_pred"] = self.out_genre(emb)
        return oup
