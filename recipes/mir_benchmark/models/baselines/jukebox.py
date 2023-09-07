import os

import jukebox
import numpy as np
import torch
from einops import rearrange
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.make_models import MODELS, make_prior, make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi

T = 8192
SAMPLE_LENGTH = 1048576  # 23.77 seconds
MODEL_PATH = "/mnt/bn/transcription/jukebox/models/5b"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/mnt/bn/transcription-49c6ce75/jukebox/models/5b"


def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)


class JukeExtractor(BaseStage):
    def __init__(
        self,
        depth=36,
        takes=["mtl_wav", "juke_wav"],
        provides=["latent"],
        serialize_opts=None,
    ):
        super().__init__(takes, provides, serialize_opts)

        # jukebox
        self.depth = depth
        self.vqvae, self.top_prior, self.hps = self.load_jukebox()
        self.get_cond()

    def load_jukebox(self):
        # Set up MPI
        rank, local_rank, device = setup_dist_from_mpi()

        # get model
        model = "5b"
        vqvae, *priors = MODELS[model]

        # set parameters
        hps = Hyperparams()
        hps.sr = 44100
        hps.n_samples = 8
        hps.name = "samples"
        hps.levels = 3
        hps.hop_fraction = [0.5, 0.5, 0.125]

        # vqvae
        hps_vqvae = setup_hparams(vqvae, dict(sample_length=SAMPLE_LENGTH))
        hps_vqvae.restore_vqvae = os.path.join(MODEL_PATH, "vqvae.pth.tar")
        vqvae = make_vqvae(hps_vqvae, device)

        # top prior language model
        hps_lm = setup_hparams(priors[-1], dict())
        hps_lm["prior_depth"] = 72
        hps_lm.restore_prior = os.path.join(MODEL_PATH, "prior_level_2.pth.tar")
        top_prior = make_prior(hps_lm, vqvae, device)
        top_prior.prior.only_encode = True

        return vqvae, top_prior, hps

    def get_cond(self):
        sample_length_in_seconds = (
            62  # model only accepts condition longer than 60 seconds
        )
        self.hps.sample_length = (
            int(sample_length_in_seconds * self.hps.sr) // self.top_prior.raw_to_tokens
        ) * self.top_prior.raw_to_tokens

        # 'lyrics' parameter is required to run the model. But it won't be actually used in our pipeline.
        metas = [
            dict(
                artist="unknown",
                genre="unknown",
                total_length=self.hps.sample_length,
                offset=0,
                lyrics="""lyrics go here!!!""",
            )
        ] * self.hps.n_samples

        labels = [None, None, self.top_prior.labeller.get_batch_labels(metas, "cuda")]
        x_cond, y_cond, prime = self.top_prior.get_cond(
            None, self.top_prior.get_y(labels[-1], 0)
        )
        self.x_single_cond = x_cond[0, :T][np.newaxis, ...]
        self.y_single_cond = y_cond[0][np.newaxis, ...]

    @torch.no_grad()
    def extract(self, audio):
        """
        Input:
            audio (torch.FloatTensor): a batch of input audio (batch, length)
        Output:
            juke_emb (torch.FloatTensor): a batch of jukebox embeddings (batch, length//128, 4800)
        """
        # always eval
        self.vqvae.eval()
        self.top_prior.eval()

        # reshape audio
        audio = audio.unsqueeze(2)  # [batch, length, 1]

        # get tokens
        zs = self.vqvae.encode(audio)
        z = zs[-1]  # [batch, length//128]

        # get conditions
        x_cond = self.x_single_cond[:, : z.shape[1]].repeat(len(z), 1, 1)
        y_cond = self.y_single_cond.repeat(len(z), 1, 1)

        # embed the tokens, and add positional embeddings and conditions.
        x = self.top_prior.prior.x_emb(z)  # token to embedding
        x = roll(x, 1)  # shift by 1 for the start token
        x[:, 0] = y_cond.view(len(z), self.top_prior.prior.width)  # fill in start token
        x = (
            x + self.top_prior.prior.pos_emb()[: z.shape[1]] + x_cond
        )  # add pos_emb and condition

        # transformer
        self.top_prior.prior.transformer.del_cache()  # clear cache
        for layer in self.top_prior.prior.transformer._attn_mods[: self.depth]:
            x = layer(x, encoder_kv=None, sample=True)
        juke_emb = x

        return juke_emb

    def forward(self, x):
        """
        Input:
            x (dict): input dictionary with two keys "mtl_wav" and "juke_wav". Each key includes torch.FloatTensor(batch, length)
        Output:
            oup (dict): output dictionary with a key "latent". Latent embedding includes torch.FloatTensor(batch, 4800, length)
        """
        # init dict
        oup = {}

        # jukebox
        x = self.extract(x["juke_wav"].squeeze(1))
        x = rearrange(x, "b t c -> b c t")

        # return dict
        oup["latent"] = x

        return oup
