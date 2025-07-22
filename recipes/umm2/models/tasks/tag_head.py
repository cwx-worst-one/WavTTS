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
        tag_types = None,
        tag_weights = None, # for karaoke only {'genres': 0.3, 'instruments': 1.0, 'tempo': 0.1, 'key': 0.1, 'year': 0.1}
        tag_type_to_id = None,   # {'genres': 0, 'instruments': 1, 'tempo': 2, 'key': 3, 'year': 4, 'GENRE': 5, 'THEME': 6, 'MOOD': 7, 'GENDER': 8, 'TIMBRE': 9, 'ARTIST': 10}
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
        self.tag_types = list(tag_types.keys())
        self.tag_type_to_id = {}
        self.tag_weights = []

        for tag_type in tag_types:
            if 'n_class' in tag_types[tag_type]:
                n_out = tag_types[tag_type]['n_class']
            else:
                n_out = len(tag_types[tag_type].keys())
            if 'weight' in tag_types[tag_type]:  # for bigmusic
                self.tag_weights.append(tag_types[tag_type]['weight'])
            elif tag_weights is not None:  # for karaoke
                self.tag_weights.append(tag_weights[tag_type])

            self.outlayers[tag_type] = nn.Sequential(
                nn.Linear(n_channel, n_hidden_channel),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_hidden_channel, n_out)
            )

            self.tag_type_to_id[tag_type] = tag_type_to_id[tag_type]
        
        
    def get_loss(self, prediction, target):
        total_loss = 0
        for weight, tag_type in zip(self.tag_weights, self.tag_types):
            if prediction[tag_type].shape[0] > 0:
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

        if "type_id" in data:
            type_id = data["type_id"]
            for tag_type in self.tag_types:
                tid = self.tag_type_to_id[tag_type]
                oup[tag_type] = oup[tag_type][type_id[:, tid]==1, :]
                                        
        output_dict = {
            "tag_pred": oup
            }

        #print([(k, labels[k].shape) for k in labels])
        if "tags" in data:
            labels = data["tags"]
            loss = self.get_loss(oup, labels)
            output_dict.update({"loss": loss, 'tags': labels})

        return output_dict
