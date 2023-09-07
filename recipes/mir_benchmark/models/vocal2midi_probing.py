from typing import Dict

from einops import rearrange
from torch import nn

from recipes.vocal2midi.models.model import GRUOutputModule
from samantha.core import BaseStage


class Vocal2midiProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        takes=["latent"],
        provides=["note", "onset"],
        resample=0,
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out_onset = nn.Sequential(
            nn.Linear(n_channel, n_hidden_channel),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(n_hidden_channel, 60),
        )
        self.out_note = nn.Linear(n_channel, 61)
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

        oup["p_vocal"], oup["o_vocal"] = self.out_note(emb), self.out_onset(emb)
        return oup


class Vocal2midiGRUStage(BaseStage):
    def __init__(
        self,
        n_channel,
        takes=["latent"],
        provides=["note", "onset"],
        resample=0,
        serialize_opts=None,
        output_dim=61,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.onset_output_module = GRUOutputModule(
            input_dim=n_channel,
            gru_hidden_dim=n_channel,
            output_dim=output_dim - 1,
            output_activation=None,
        )
        self.note_output_module = GRUOutputModule(
            input_dim=n_channel,
            gru_hidden_dim=n_channel,
            context_dim=output_dim - 1,
            output_dim=output_dim,
            output_activation=None,
        )

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
        emb = rearrange(emb, "b c t -> b t c")
        onset_offset = self.onset_output_module(emb)
        note = self.note_output_module(emb, onset_offset.detach())

        oup["p_vocal"], oup["o_vocal"] = note, onset_offset
        return oup
