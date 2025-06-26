from torch import nn
from typing import Dict
from einops import rearrange

from samantha.core import BaseStage

class TaggingProbingStage(BaseStage):
    def __init__(
        self,
        n_channel,
        n_hidden_channel=512,
        # n_out=21,
        takes=["latent"],
        provides=["tag_pred"],
        serialize_opts=None,
        one_shot_map = None,
        tag_types = None,
        post_pool=False,
        final_pooling=True,
        *args,
        **kwargs,
    ):
        super().__init__(takes, provides, serialize_opts)
        self.out = {}
        self.tag_types = tag_types
        self.post_pool = post_pool
        self.final_pooling = final_pooling
        self.outlayers = nn.ModuleDict()
        for tag_type in self.tag_types:
            n_out = len(one_shot_map[tag_type].keys())
            self.outlayers[tag_type] = nn.Sequential(
                nn.Linear(n_channel, n_hidden_channel),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_hidden_channel, n_out)
            )
        
    def forward(self, data: Dict) -> Dict:
        """
        Input:
            data (dict): input dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, channels, length)
        Output:
            oup (dict): output dictionary with a key "tag_pred".
            oup["genre_pred"] (list): a list includes torch.FloatTensor(batch, class)
        """            
        # init dict
        oup = {}
        
        if self.post_pool == False:
            emb = data["latent"].mean(-1)
        else:
            emb = data["latent"]
            emb = rearrange(emb, "b c t -> b t c")

        for tag_type in self.tag_types:
            if self.post_pool == False:
                oup[tag_type] = self.outlayers[tag_type](emb)
            else:
                embedding = self.outlayers[tag_type](emb)
                if self.final_pooling:
                    oup[tag_type] = rearrange(embedding, "b t c -> b c t").mean(-1)
                else:
                    oup[tag_type] = rearrange(embedding, "b t c -> b c t")

        return {'tag_pred': oup}
