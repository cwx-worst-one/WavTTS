import torch
from einops import rearrange
from torch import nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class MERT(nn.Module):
    def __init__(self):
        super(MERT, self).__init__()

        # processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )

        # MERT
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )

    def get_latent(self, audio, layer_ix):
        # wav2vec
        inp = self.processor(audio, sampling_rate=24000, return_tensors="pt")
        inp["input_values"] = inp["input_values"][0].to(audio.device)
        inp["attention_mask"] = torch.ones(
            inp["input_values"].size(), dtype=torch.int32
        ).to(audio.device)

        emb = self.mert(**inp, output_hidden_states=True)["hidden_states"][layer_ix]
        return emb
