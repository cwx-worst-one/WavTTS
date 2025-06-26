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
        tag_weights = None, # for karaoke {'genres': 0.3, 'instruments': 1.0, 'tempo': 0.1, 'key': 0.1, 'year': 0.1}
        tag_type_to_dataset = None,   # {'GENRE': 0, 'THEME': 0, 'MOOD': 0, 'GENDER': 0, 'TIMBRE': 0, 'genres': 1, 'instruments': 1, 'tempo': 1, 'key': 1, 'year': 1}
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
        self.tag_type_to_dataset = tag_type_to_dataset
        self.tag_weights = []

        for tag_type in self.tag_types:
            if 'n_class' in self.tag_types[tag_type]:
                n_out = self.tag_types[tag_type]['n_class']
            else:
                n_out = len(tag_types[tag_type].keys())
            if 'weight' in self.tag_types[tag_type]:
                self.tag_weights.append(self.tag_types[tag_type]['weight'])
            elif tag_weights is not None:
                self.tag_weights.append(tag_weights[tag_type])

            self.outlayers[tag_type] = nn.Sequential(
                nn.Linear(n_channel, n_hidden_channel),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_hidden_channel, n_out)
            )
        
        
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

        if "dataset_id" in data:
            dataset_id = data["dataset_id"]
            for tag_type in self.tag_types:
                did = self.tag_type_to_dataset[tag_type]
                oup[tag_type] = oup[tag_type][dataset_id==did, :]
                                        
        output_dict = {
            "tag_pred": oup
            }

        #print([(k, labels[k].shape) for k in labels])
        if "tags" in data:
            labels = data["tags"]
            loss = self.get_loss(oup, labels)
            output_dict.update({"loss": loss, 'tags': labels})

        return output_dict
