from torch import nn
from typing import Dict
from einops import rearrange

from recipes.umm2.models.base import BaseStage

class TaggingProbingStage(BaseStage):
    def __init__(
        self,
        task,
        n_channel,
        n_hidden_channel=512,
        takes=["latent"],
        provides=["tag_pred"],
        bypasses=[],
        tag_map = None,
        tag_types = None,
        tag_weights = None,
        final_pooling=True,
        lr_ratio = 1.0,
        loss_weight = 1.0,
        is_frozen = False,
    ):
        BaseStage.__init__(self, takes, provides, bypasses, task=task, lr_ratio=lr_ratio, loss_weight=loss_weight, is_frozen=is_frozen)

        self.out = {}
        self.final_pooling = final_pooling
        self.outlayers = nn.ModuleDict()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.tag_types = tag_types
        if isinstance(self.tag_types, list):
            self.tag_weights = [1.0] * len(tag_types) if tag_weights is None else tag_weights
            for tag_type in self.tag_types:
                n_out = len(tag_map[tag_type].keys())
                self.outlayers[tag_type] = nn.Sequential(
                    nn.Linear(n_channel, n_hidden_channel),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(n_hidden_channel, n_out)
                )
        elif isinstance(self.tag_types, dict):
            self.tag_weights = []
            for tag_type in self.tag_types:
                self.tag_weights.append(self.tag_types[tag_type]['weight'])
                n_out = self.tag_types[tag_type]['n_class']
                self.outlayers[tag_type] = nn.Sequential(
                    nn.Linear(n_channel, n_hidden_channel),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(n_hidden_channel, n_out)
                )
        
        
    def get_loss(self, prediction, target):
        total_loss = 0
        for weight, tag_type in zip(self.tag_weights, self.tag_types):
            loss = self.loss_fn(prediction[tag_type], target[tag_type])
            total_loss += weight * loss
        return total_loss

    def _compute(self, data: Dict) -> Dict:
        """
        Input:
            data (dict): input dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, channels, length)
        Output:
            oup (dict): output dictionary with a key "tag_pred".
            oup["genre_pred"] (list): a list includes torch.FloatTensor(batch, class)
        """            
        # init dict
        oup = {}     
        emb = data["latent"]

        for tag_type in self.tag_types:
            embedding = self.outlayers[tag_type](emb)
            if self.final_pooling:
                oup[tag_type] = rearrange(embedding, "b t c -> b c t").mean(-1)
            else:
                oup[tag_type] = rearrange(embedding, "b t c -> b c t")

        output_dict = {
            "tag_pred": oup
            }

        #print([(k, labels[k].shape) for k in labels])
        if "tags" in data:
            labels = data["tags"]
            loss = self.get_loss(oup, labels)
            output_dict.update({"loss": loss, 'tags': labels})

        return output_dict
