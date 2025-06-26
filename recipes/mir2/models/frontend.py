import warnings
import torch

from einops import rearrange
from samantha.core import BaseStage


class Frontend(BaseStage):
    def __init__(
            self, 
            model,
            takes=["audio"],
            provides=["latent"],
            serialize_opts=None,
            layer_ix=12,
            is_update=False,
            ):
        super().__init__(takes, provides, serialize_opts)

        self.model = model
        self.takes = takes
        self.layer_ix = layer_ix
        self.is_update = is_update
        if not is_update:
            self.model.eval()

    def extract(self, audio):
        """
        Input:
            audio (torch.FloatTensor): a batch of input audio (batch, length)
        Output:
            emb (torch.FloatTensor): a batch of output embeddings (batch, length, 1024)
        """

        if len(audio.shape) == 2:
            audio = rearrange(audio, "b t -> b 1 t")
 
        if self.is_update:
            emb = self.model.get_latent(audio, self.layer_ix)
        else:
            self.model.eval()
            with torch.no_grad():
                emb = self.model.get_latent(audio, self.layer_ix)

        return emb

    def forward(self, x):
        """
        Input:
            x (dict): input dictionary with a key "audio". Input includes torch.FloatTensor(batch, length)
        Output:
            oup (dict): output dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, 1024, length)
        """            
        # init dict
        oup = {}

        # BEST-RQ]\

        x = self.extract(x[self.takes[0]])

        '''
        except RuntimeError as e:
            if x[self.takes[0]].shape[0] == 1:
                warnings.warn("Torchscript only supports a batch size larger than 1.")
                x = self.extract(x[self.takes[0]].repeat(2, 1))
                x = x[:1]
            else: 
                raise e
        '''

        x = rearrange(x, "b t c -> b c t")
    
        # return dict
        oup["latent"] = x

        return oup


class FrontendMulti(BaseStage):
    def __init__(
            self, 
            model,
            takes=["audio"],
            provides=["latent", "loss", "acc"],
            serialize_opts=None,
            ):
        super().__init__(takes, provides, serialize_opts)

        self.model = model
        self.takes = takes

    def extract(self, audio):
        """
        Input:
            audio (torch.FloatTensor): a batch of input audio (batch, length)
        Output:
            emb (torch.FloatTensor): a batch of output embeddings (batch, length, 1024)
        """

        if self.is_update:
            emb = self.model.get_latent(audio, self.layer_ix)
        else:
            self.model.eval()
            with torch.no_grad():
                emb = self.model.get_latent(audio, self.layer_ix)

        return emb

    def forward(self, x):
        """
        Input:
            x (dict): input dictionary with a key "audio". Input includes torch.FloatTensor(batch, length)
        Output:
            oup (dict): output dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, 1024, length)
        """            
        # init dict
        oup = {}

        # BEST-RQ
        logits, hidden_emb, losses, accuracies = self.model(x[self.takes[0]])

        # return dict
        oup["latent"] = rearrange(hidden_emb[-1], "b t c -> b c t")
        oup["loss"] = losses
        oup["acc"] = accuracies

        return oup