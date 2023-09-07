from typing import Dict

from einops import rearrange
from torch import nn

from samantha.core import BaseStage


class ChordProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        takes=["latent"],
        provides=["chord_root", "chord_triad"],
        resample=0,
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_root = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden_channel, 13),
        )
        self.out_triad = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden_channel, 7),
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
            oup (dict): output dictionary with two keys "chord_root" and "chord_triad".
            oup["chord_root"] (torch.FloatTensor): predicted chord root (batch, length, 13)
            oup["chord_triad"] (torch.FloatTensor): predicted chord triad (batch, length, 7)
        """
        oup = {}
        emb = data["latent"]
        if self.is_resample:
            emb = self.avg_pool(emb)
        emb = rearrange(emb, "b c t -> b t c")
        oup["chord_root"], oup["chord_triad"] = self.out_root(emb), self.out_triad(emb)
        return oup
