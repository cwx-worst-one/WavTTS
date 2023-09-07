from typing import Dict

import torch
from einops import rearrange
from torch import nn

from samantha.core import BaseStage


class GRUOutputModule(nn.Module):
    def __init__(
        self,
        gru_layers=2,
        input_dim=481,
        context_dim=0,
        gru_hidden_dim=481,
        output_dim=481,
        input_ln=True,
        output_activation="softmax",
    ):
        super(GRUOutputModule, self).__init__()
        self.input_ln = input_ln

        if input_ln:
            self.ln_gru = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim + context_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Sequential(
            nn.LayerNorm(2 * gru_hidden_dim), nn.Linear(2 * gru_hidden_dim, output_dim)
        )
        self.relu = nn.ReLU()

        if output_activation == "softmax":
            self.output_activation = nn.Softmax(dim=-1)
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation is None:
            self.output_activation = nn.Identity()

    def forward(self, x, context=None):
        if self.input_ln:
            x = self.ln_gru(x)

        if context is not None:
            x = self.gru(torch.cat([x, context], dim=-1))[0]
        else:
            x = self.gru(x)[0]

        out = self.linear(x)
        out = self.output_activation(out)
        return out


class TaggingInstrumentGRUStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        n_out=17,
        takes=["latent"],
        provides=["genre_pred"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.gru = GRUOutputModule(
            gru_layers=2,
            input_dim=n_channel,
            gru_hidden_dim=n_hidden_channel,
            output_dim=n_out,
            output_activation="sigmoid",
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

        # gru
        emb = rearrange(data["latent"], "b c t -> b t c")
        oup["instrument_pred"] = self.gru(emb)[:, -1, :]

        return oup


class TaggingGenreGRUStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        n_out=34,
        takes=["latent"],
        provides=["genre_pred"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.gru = GRUOutputModule(
            gru_layers=2,
            input_dim=n_channel,
            gru_hidden_dim=n_hidden_channel,
            output_dim=n_out,
            output_activation="sigmoid",
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

        # gru
        emb = rearrange(data["latent"], "b c t -> b t c")
        oup["genre_pred"] = self.gru(emb)[:, -1, :]

        return oup


class TaggingVocalGRUStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        n_out=16,
        takes=["latent"],
        provides=["vocal_pred"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.gru = GRUOutputModule(
            gru_layers=2,
            input_dim=n_channel,
            gru_hidden_dim=n_hidden_channel,
            output_dim=n_out,
            output_activation="sigmoid",
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

        # gru
        emb = rearrange(data["latent"], "b c t -> b t c")
        oup["vocal_pred"] = self.gru(emb)[:, -1, :]

        return oup


class TaggingMultiGRUStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        n_out=67,
        takes=["latent"],
        provides=["instrument_pred", "vocal_pred", "genre_pred"],
        serialize_opts=None,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.gru = GRUOutputModule(
            gru_layers=2,
            input_dim=n_channel,
            gru_hidden_dim=n_hidden_channel,
            output_dim=n_out,
            output_activation="sigmoid",
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

        # gru
        emb = rearrange(data["latent"], "b c t -> b t c")
        out = self.gru(emb)[:, -1, :]
        oup["instrument_pred"] = out[:, :17]
        oup["vocal_pred"] = out[:, 17:33]
        oup["genre_pred"] = out[:, 33:67]

        return oup
