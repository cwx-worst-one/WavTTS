import torch
import torch.nn as nn
import torch.nn.functional as F

from recipes.soundstream.models.modules.ema_vqvae import EMAVectorQuantizer
from recipes.soundstream.models.modules.encoder import Encoder
from recipes.soundstream.models.modules.generator import Generator
from recipes.soundstream.models.modules.vqvae import VectorQuantizer


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


if __name__ == "__main__":

    model = VQGAN(
        model_type="bytewave_wn_causal",
        num_res=12,
        quant_token_num=1024,
        quant_token_dim=256,
        quant_beta=0.25,
        down_rates=[2, 2, 10, 12],
        upsample_rates=[10, 6, 4, 2],
        encoder_initial_channel=16,
        decoder_initial_channel=768,
        trunc_noise=False,
        smaller_encoder=True,
        init_cluster_size=32,
    )
    x = torch.randn(size=[2, 1, 32 * 600])
    decoder_out, quant_loss, quant_index, encoder_out = model(x)
    print(
        f"Input shape: {x.shape}\ndecoder_out shape: {decoder_out.shape}\nencoder_out shape: {encoder_out.shape}"
    )

    # err = (decoder_out - 1).mean() + quant_loss
    # err.backward()
    # for name, params in model.named_parameters():
    #     if params.grad is None:
    #         print(name, params.shape)
