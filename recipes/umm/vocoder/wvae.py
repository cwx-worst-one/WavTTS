import torch
import yaml
import torch.nn.functional as F

from torch import nn
from torch.nn.utils import weight_norm
from easydict import EasyDict

from recipes.umm.vocoder.flow import ResidualCouplingBlock, WN
from recipes.umm.vocoder.bigvgan import BigVGAN
from recipes.umm.vocoder.llama_v100 import LLaMa, ModelArgs
from recipes.umm.vocoder.ctiga_llama import CtigaLLaMa
from recipes.umm.vocoder.audio_encoder import AudioEncoder, SpecEncoder, MelEncoder, MelPredictor
from recipes.umm.vocoder.losses import kl_loss_vits


def calculate_model_params(model, model_name):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params of \"{}\": {}, size of saving: {}M".format(
        model_name,
        pytorch_total_params,
        pytorch_total_params*4/1024/1024))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params of \"{}\": {}, size of saving: {}M".format(
        model_name,
        pytorch_total_params,
        pytorch_total_params*4/1024/1024))
    return


class Transpose(nn.Module):
    def __init__(self, dim1=-1, dim2=-2):
        super().__init__()

    def forward(self, x):
        return x.transpose(-1, -2)


class VAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.act = torch.nn.GELU()
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels * 2, kernel_size=1, bias=False))
        self.bn = torch.nn.BatchNorm1d(out_channels, affine=False, momentum=0.01)
        self.m_scale = nn.Parameter(torch.ones([1, out_channels, 1]))

    def forward(self, x):
        x = self.conv(self.act(x))
        m, logs = torch.chunk(x, chunks=2, dim=1)
        m = self.bn(m) * self.m_scale
        z = m + torch.randn_like(m) * logs.exp()
        return z, m, logs


class WaveformVAE(nn.Module):
    def __init__(self, hp, sd_channels, hidden_channels):
        super().__init__()
        self.audio_encoder = AudioEncoder(hidden_channels) # 25Hz
        self.spec_encoder = SpecEncoder(hidden_channels)
        self.mel_encoder = MelEncoder(hidden_channels)
        self.weight_sum = nn.Parameter(torch.zeros([hidden_channels, 3, 1]))
        # GPT
        llama_config = ModelArgs(out_dim=1024)
        self.llama = nn.Sequential(
            Transpose(),
            nn.Linear(hidden_channels, llama_config.out_dim, bias=False),
            # LLaMa(llama_config, token_input=False),
            CtigaLLaMa(llama_config),
            Transpose(),
        )
        self.vae = VAE(in_channels=llama_config.out_dim, out_channels=sd_channels)
        self.flow = ResidualCouplingBlock(
            sd_channels,
            hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            affine=False,
        )
        self.mel_predictor = MelPredictor(
            in_channels=sd_channels,
            out_channels=120,
            hidden_channels=256,
            n_layers=4,
            p_dropout=0,
        )
        self.hp = EasyDict(hp)
        self.vocoder = BigVGAN(self.hp)

        calculate_model_params(self.audio_encoder, "audio_encoder")
        calculate_model_params(self.spec_encoder, "spec_encoder")
        calculate_model_params(self.mel_encoder, "mel_encoder")
        calculate_model_params(self.llama, "llama")
        calculate_model_params(self.flow, "flow")
        calculate_model_params(self.vocoder, "vocoder")

    def forward(self, input_dict):
        wav = input_dict["audio"] # [b, t]
        audio_feature = self.audio_encoder(wav.unsqueeze(1)) # [b, d, t]
        spec_feature, gt_spec = self.spec_encoder(wav)
        mel_feature, gt_mel = self.mel_encoder(wav)
        # weight sum of features
        feature = torch.stack([audio_feature, spec_feature, mel_feature], dim=2) # [b, d, 3, t]
        feature = (self.weight_sum.softmax(dim=1) * feature).sum(dim=2)
        # GPT enhance !!!
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            feature = self.llama(feature)
        z, m, logs = self.vae(feature.float())
        z_flow, log_det = self.flow(z)

        # mel predict
        mel_z = self.mel_predictor(z)

        # vocoder
        vocoder_in, wav_in = self.slice_feature_wav(z, wav)
        vocoder_out = self.vocoder(vocoder_in)

        # z_p, logs_q, m_p, logs_p, z_mask
        kl_loss = kl_loss_vits(z_flow, logs, torch.zeros_like(logs), torch.zeros_like(logs), torch.ones_like(logs))
        output_dict = {
            "gt_wav": wav_in.unsqueeze(1),
            "gen_wav": vocoder_out,
            "z": z,
            "m": m,
            "logs": logs,
            "z_flow": z_flow,
            "mel_z": mel_z,
            "kl_loss": kl_loss,
            "gt_spec": gt_spec,
            "gt_mel": gt_mel,
        }
        return output_dict

    def infer_from_z(self, z):
        vocoder_out = self.vocoder(vocoder_in)
        return vocoder_out

    def get_z_m_logs(self, wav):
        audio_feature = self.audio_encoder(wav.unsqueeze(1)) # [b, d, t]
        spec_feature, gt_spec = self.spec_encoder(wav)
        mel_feature, gt_mel = self.mel_encoder(wav)
        # weight sum of features
        feature = torch.stack([audio_feature, spec_feature, mel_feature], dim=2) # [b, d, 3, t]
        feature = (self.weight_sum.softmax(dim=1) * feature).sum(dim=2)
        # GPT enhance !!!
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            feature = self.llama(feature)
        z, m, logs = self.vae(feature.float())
        return z, m, logs

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def slice_feature_wav(self, feature, wav):
        segment_size = self.hp.segment_size
        feature_size = segment_size // self.hp.hop_size
        # prevent OOM
        max_bs = self.hp.max_bs
        bs = feature.shape[0]
        rand_beg = torch.randint(low=0, high=feature.shape[-1] - feature_size + 1, size=[max_bs,]).to(wav.device)
        slice_features = []
        slice_wavs = []
        for i, beg in enumerate(rand_beg):
            slice_features.append(feature[i % bs, :, beg:beg + feature_size])
            slice_wavs.append(wav[i % bs, beg * self.hp.hop_size:beg * self.hp.hop_size + segment_size])
        slice_features = torch.stack(slice_features, dim=0)
        slice_wavs = torch.stack(slice_wavs, dim=0)

        return slice_features, slice_wavs

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def preprocessing(self, audio):
        assert audio.dtype == torch.float32
        if audio.shape[-1] < self.hp.segment_size:
            audio = F.pad(audio, (0, self.hp.segment_size - audio.shape[-1]))
        return audio

    
if __name__ == '__main__':
    wvae = WaveformVAE(64, 256).cuda()
    input_dict = {"audio": (torch.randn(size=[1, 24000 * 10]) / 6).cuda()}
    out_dict = wvae(input_dict)
    for key in out_dict:
        print(key, out_dict[key].shape)

    '''
    def backward_receptive_field(model, dim=1, length=24000 * 10, device='cpu'):
        model = model.to(device).eval()

        x = torch.randn(size=[1, dim, length]).to(device)
        x.requires_grad_(True)
        x.retain_grad()

        # Make a forward pass
        out = model({"audio": x[:, 0, :]})['gen_wav']
        # Create gradient variable
        grad = torch.zeros_like(out)
        grad[:, :, grad.shape[-1] // 2] = 1

        # Make a backward pass
        out.backward(grad)

        # Check non-zero values
        gradmap = x.grad.squeeze(0)
        gradmap = (gradmap != 0).sum(0)  # sum across features
        print(torch.nonzero(gradmap).squeeze().detach().cpu().numpy())
        rf = (gradmap != 0).sum()
        print("Backward receptive field: {}".format(rf))
        return
    backward_receptive_field(wvae, device='cuda')
    '''