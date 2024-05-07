import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from recipes.soundstream.models.modules.ema_vqvae import EMAVectorQuantizer
from recipes.soundstream.models.modules.encoder import Encoder
from recipes.soundstream.models.modules.generator import Generator
from recipes.soundstream.models.modules.vqvae import VectorQuantizer

from recipes.soundstream.models.modules.encoder_new import Encoder as Encoder_new
from recipes.soundstream.models.modules.decoder import Decoder

class VQGAN(nn.Module):
    def __init__(
        self,
        model_type,
        num_res,
        quant_token_num,
        quant_token_dim,
        quant_beta,
        down_rates,
        upsample_rates,
        encoder_initial_channel,
        decoder_initial_channel,
        trunc_noise,
        smaller_encoder,
        init_cluster_size=1,
        dist=True,
    ):
        super().__init__()
        self.encoder = Encoder(
            down_rates=down_rates,
            encoder_initial_channel=encoder_initial_channel,
            trunc_noise=trunc_noise,
            smaller_encoder=smaller_encoder,
            model_type=model_type,
        )
        self.num_res = num_res
        self.quant_vaes = nn.ModuleList()
        if not isinstance(quant_token_num, (tuple, list)):
            quant_token_nums = [quant_token_num] * num_res
        else:
            quant_token_nums = quant_token_num
        for i in range(self.num_res):
            self.quant_vaes.append(
                EMAVectorQuantizer(
                    quant_token_num=quant_token_nums[i],
                    quant_token_dim=quant_token_dim,
                    quant_beta=quant_beta,
                    init_cluster_size=init_cluster_size,
                    dist=dist,
                )
            )
        self.decoder = Generator(
            upsample_rates=upsample_rates,
            decoder_initial_channel=decoder_initial_channel,
            encoder_initial_channel=encoder_initial_channel,
            model_type=model_type,
            trunc_noise=trunc_noise,
        )

    def forward(self, x, warmup=False):
        encoder_out = self.encode(x)
        quant_out, quant_loss, quant_index = self.quant(encoder_out)
        decoder_out = self.decode(quant_out)
        return decoder_out, quant_loss, quant_index, encoder_out

    def encode(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def quant(self, x, warmup=False):
        encoder_out = x
        quant_outs = []
        losses = []
        quant_indexs = []

        for quant_vae in self.quant_vaes:
            quant_out, loss, quant_index = quant_vae(encoder_out)
            quant_outs.append(quant_out)
            losses.append(loss)
            quant_indexs.append(quant_index)
            encoder_out = encoder_out - quant_out.detach()

        quant_out = sum(quant_outs)
        quant_loss = sum(losses)
        quant_index = quant_indexs
        return quant_out, quant_loss, quant_index

    def decode(self, x):
        decoder_out = self.decoder(x)
        return decoder_out

    def get_quant_output_from_index(self, index):
        # index: should be list/tuple or tensor
        # if index is a list/tuple, each tensor's shape must be [b, t]
        # if index is a tensor, the tensor's shape must be [b, n_codebook, t]
        quant_outs = []
        if isinstance(index, (tuple, list)):
            for i, ind in enumerate(index):
                embedding = self.quant_vaes[i].embedding(ind)  # [b, t, d]
                quant_outs.append(embedding.transpose(1, 2))  # [b, d, t]
        elif isinstance(index, torch.Tensor):
            for i in range(index.size(1)):
                ind = index[:, i, :]
                embedding = self.quant_vaes[i].embedding(ind)  # [b, t, d]
                quant_outs.append(embedding.transpose(1, 2))  # [b, d, t]
        else:
            print("Unexpected index type: {}".format(type(index)))
            raise Exception

        quant_out = sum(quant_outs)
        return quant_out

class VQGAN_KL(nn.Module):
    def __init__(
        self,
        model_type,
        quant_token_dim,
        down_rates,
        upsample_rates,
        encoder_initial_channel,
        decoder_initial_channel,
        trunc_noise,
        smaller_encoder,
        init_cluster_size=1,
        dist=True,
    ):
        super().__init__()
        self.encoder = Encoder(
            down_rates=down_rates,
            encoder_initial_channel=encoder_initial_channel,
            trunc_noise=trunc_noise,
            smaller_encoder=smaller_encoder,
            model_type=model_type,
        )
        self.mean_logvar_conv = nn.Conv1d(quant_token_dim, 32, 1)

        self.decoder = Generator(
            upsample_rates=upsample_rates,
            decoder_initial_channel=decoder_initial_channel,
            encoder_initial_channel=encoder_initial_channel,
            model_type=model_type,
            trunc_noise=trunc_noise,
        )

    def forward(self, x, warmup=False, deterministic=False):
        encoder_out = self.encode(x)
        sample, kl_loss, std_mean = self.sample(encoder_out, deterministic=deterministic)
        decoder_out = self.decode(sample)
        return decoder_out, kl_loss, std_mean

    def encode(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def sample(self, x, deterministic=False):
        mean_logvar = self.mean_logvar_conv(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            kl_loss = torch.FloatTensor([0.0]).to(x.device)
            sample = mean
        else:
            kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2])
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        return sample, kl_loss.mean(), std.mean()


    def decode(self, x):
        decoder_out = self.decoder(x)
        return decoder_out

class VQGAN_KL_new(nn.Module):
    def __init__(
            self,
            n_channels=1,
            latent_dim=128,
            downsample_rates=[2, 3, 7, 10],
            upsample_rates=[10, 7, 3 ,2],
            encoder_base_dim=96,
            decoder_base_dim=2560,
            adapt_hopper=True,
            last_act=True
        ):
        super().__init__()
        self.encoder = Encoder_new(
            n_channels=n_channels,
            d_model=encoder_base_dim,
            strides=downsample_rates,
            adapt_hopper=adapt_hopper,
        )

        self.mean_logvar_conv = nn.Conv1d(self.encoder.enc_dim, latent_dim*2, 1)
        self.latent_drop = nn.Dropout(0.05)
        self.decoder = Decoder(
            input_channel=latent_dim,
            channels=decoder_base_dim,
            rates=upsample_rates,
            d_out=n_channels,
            adapt_hopper=adapt_hopper,
            last_act=last_act
        )
   
    def forward(self, x,  deterministic=False):
        encoder_out = self.encode(x)
        sample, kl_loss, std_mean = self.sample(encoder_out, deterministic=deterministic)
        sample = self.latent_drop(sample)
        decoder_out = self.decode(sample)
        return decoder_out, kl_loss, std_mean

    def encode(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def sample(self, x, deterministic=False):
        mean_logvar = self.mean_logvar_conv(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            kl_loss = torch.FloatTensor([0.0]).to(x.device)
            sample = mean
        else:
            kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2])
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        return sample, kl_loss.mean(), std.mean()


    def decode(self, x):
        decoder_out = self.decoder(x)
        return decoder_out

class VQGAN_KL_mix(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=2,
            latent_dim=128,
            downsample_rates=[2, 3, 7, 10],
            upsample_rates=[10, 7, 3 ,2],
            encoder_base_dim=96,
            decoder_base_dim=2560,
            adapt_hopper=True,
            ckpt_path=None,
        ):
        super().__init__()
        self.encoder = Encoder_new(
            n_channels=in_channels,
            d_model=encoder_base_dim,
            strides=downsample_rates,
            adapt_hopper=adapt_hopper,
        )
        self.mean_logvar_conv = nn.Conv1d(self.encoder.enc_dim, latent_dim * 2, 1)
        self.latent_drop = nn.Dropout(0.05)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            new_dict = OrderedDict()
            for key in ckpt['state_dict']:
                if key.endswith("total_ops") or key.endswith("total_params"):
                    continue
                if 'encoder' in key:
                    new_dict[key.replace('generator.encoder.', '')] = ckpt['state_dict'][key]
            self.encoder.load_state_dict(new_dict)

            new_dict = OrderedDict()
            for key in ckpt['state_dict']:
                if key.endswith("total_ops") or key.endswith("total_params"):
                    continue
                if 'mean_logvar_conv' in key:
                    new_dict[key.replace('generator.mean_logvar_conv.', '')] = ckpt['state_dict'][key]
            self.mean_logvar_conv.load_state_dict(new_dict)

        self.decoder = Decoder(
            input_channel=latent_dim,
            channels=decoder_base_dim,
            rates=upsample_rates,
            d_out=out_channels,
            adapt_hopper=adapt_hopper,
        )
   
    def forward(self, x,  deterministic=False):
        self.encoder.eval()
        self.mean_logvar_conv.eval()
        with torch.no_grad():
            encoder_out = self.encode(x)
            sample, kl_loss, std_mean = self.sample(encoder_out, deterministic=deterministic)
        decoder_out = self.decode(sample.detach())
        return decoder_out, kl_loss, std_mean

    def encode(self, x):
        encoder_out = self.encoder(x)
        return encoder_out

    def sample(self, x, deterministic=False):
        mean_logvar = self.mean_logvar_conv(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        if deterministic:
            kl_loss = torch.FloatTensor([0.0]).to(x.device)
            sample = mean
        else:
            kl_loss = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar, dim=[1, 2])
            sample = mean + std * torch.randn_like(mean).to(device=x.device)
        sample = sample.clamp(-3, 3) / 3
        return sample, kl_loss.mean(), std.mean()


    def decode(self, x):
        decoder_out = self.decoder(x)
        return decoder_out
 
if __name__ == "__main__":

    # model = VQGAN_KL_new(
    #     n_channels=1,
    #     latent_dim=128,
    #     downsample_rates=[2, 3, 7, 10],
    #     upsample_rates=[10, 7, 3 ,2],
    #     encoder_base_dim=64,
    #     decoder_base_dim=64,
    # )
    # # 44100= 2*2*3*3*5*5*7*7
    # x = torch.randn(size=[2, 1, int(25200)])
    # decoder_out, kl_loss, std_mean = model(x)
    # print(
    #     f"Input shape: {x.shape}\ndecoder_out shape: {decoder_out.shape}\nkl_loss shape: {kl_loss.shape}\nstd_mean shape: {std_mean.shape}"
    # )
    # ckpt = torch.load('soundstream-step=1240000-val_sdr=12.5666-EMA.ckpt', map_location='cpu')
    # print(ckpt['state_dict'].keys())

    # assert 1==2
    model = VQGAN_KL_new(
        n_channels=2,
        latent_dim=32,
        downsample_rates=[10, 9, 5, 2],
        upsample_rates= [2, 5, 9 ,10],
        encoder_base_dim=96,
        decoder_base_dim=2560,
        adapt_hopper=True,
        # ckpt_path=None,
    )
    # 352 = 2**5 * 11
    x = torch.randn(size=[2, 2, int(18000)])
    decoder_out, kl_loss, std_mean = model(x)
    print(
        f"Input shape: {x.shape}\ndecoder_out shape: {decoder_out.shape}"
    )

