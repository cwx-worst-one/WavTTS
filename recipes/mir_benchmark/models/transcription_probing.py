from typing import Dict

from einops import rearrange
from torch import nn

from samantha.core import BaseStage


class TranscriptionProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        takes=["latent"],
        provides=["note", "onset"],
        resample=0,
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_onset = nn.Linear(n_channel, 12 * 128)
        self.out_note = nn.Linear(n_channel, 12 * 128)
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
        note = self.out_note(emb)
        onset = self.out_onset(emb)
        b, t, f = note.shape
        note = note.reshape(b, t, 12, -1).permute(0, 2, 1, 3)
        onset = onset.reshape(b, t, 12, -1).permute(0, 2, 1, 3)

        oup["note"], oup["onset"] = note, onset
        return oup
