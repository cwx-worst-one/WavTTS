import json
import torch
import torchaudio
import numpy as np
from torch import nn
from einops import rearrange
from transformers import AutoModel


from recipes.musicfm.modules.random_quantizer import RandomProjectionQuantizer
from recipes.musicfm.modules.features import MelSTFT
from recipes.musicfm.modules.conv import Conv2dSubsampling


def reset_parameters(model):
    for module in model.children():
        if isinstance(module, nn.Module):
            reset_parameters(module)
    
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()


class MERT_RQ(nn.Module):
    """ 
    MusicFM

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 24-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        codebook_dim=16,
        codebook_size=8192,
        hop_length=160,
        n_fft=2048,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=24,
        mask_hop=0.4,
        mask_prob=0.6,
        is_flash=True,
        stat_path=None,
        model_path=None,
        ):
        super(MERT_RQ, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.codebook_size = codebook_size
        self.features = ["melspec"]
        
        # load feature mean / std stats
        with open(stat_path, "r") as f:
            self.stat = json.load(f)

        # multiple random quantizers
        self.quantizer_melspec = RandomProjectionQuantizer(n_mels * 2, codebook_dim, codebook_size)  # mel spec

        # feature extractor
        self.preprocessor_melspec = MelSTFT(n_fft=n_fft, hop_length=hop_length)

        # load mert
        self.mert = self.get_mert()

        # Conformer
        if is_flash:
            self.mert = self.mert.half()

        # projection
        self.linear = nn.Linear(encoder_dim, codebook_size)
        
        # loss function
        self.loss = nn.CrossEntropyLoss()

        # load model
        if model_path:
            S = torch.load(model_path)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            self.load_state_dict(SS, strict=False)

    def get_mert(self):
        mert = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        reset_parameters(mert)
        return mert

    def masking(self, x):
        """ random masking of 400ms with given probability """
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop)
        len_masking_token = int(24000/self.hop_length/2/2 * self.mask_hop)

        # get random mask indices
        start_indices = torch.rand(b, t//len_masking_raw-1) < self.mask_prob
        time_domain_masked_indices = torch.nonzero(start_indices.repeat_interleave(len_masking_raw, dim=1))
        token_domain_masked_indices = torch.nonzero(start_indices.repeat_interleave(len_masking_token, dim=1))

        # mask with random values
        masking_noise = torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1 # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)
        
        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x, features):
        """ extract classic audio features """
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    def encoder(self, x):
        """ 2-layer conv + w2v-conformer """
        x = self.conv(x)
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i*self.codebook_size:(i+1)*self.codebook_size] 
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb

    @torch.no_grad()
    def normalize(self, x):
        """ normalize the input audio to have zero mean unit variance """
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key]
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """ rearrange the batch to flatten every 2 steps """
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=2)[:, :-1, :]
        return x

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for key in x.keys():
            layer = getattr(self, "quantizer_%s" % key)
            out[key] = layer(x[key])
        return out

    def get_targets(self, x):
        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)
        x = self.rearrange(x)
        target_tokens = self.tokenize(x)
        return target_tokens

    def get_predictions(self, x):
        # mert
        last_emb = self.mert(x.squeeze(1))["last_hidden_state"]
        
        # encoding
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i*self.codebook_size:(i+1)*self.codebook_size] 
            for i, key in enumerate(self.features)
        }

        return logits, 0

    def get_metrics(self, x):
        x = 0

    def get_latent(self, x, layer_ix=12):
        _, hidden_states = self.get_predictions(x)
        emb = hidden_states[layer_ix]
        return emb


    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            masked_logits = logits[key][tuple(masked_indices.t())]
            masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = torch.sum(masked_logits.argmax(-1) == masked_tokens) / masked_tokens.numel()
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x)

        # masking
        x, masked_indices = self.masking(x)
            
        # forward
        logits, hidden_emb = self.get_predictions(x)

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)
        print(losses, accuracies)

        return logits, hidden_emb, losses, accuracies

        
