import torch
import torch.nn.functional as F
from torch import nn
from recipes.umm.models.dualumm_encoders import ConvStacksWithDownUpSampling
from transformers.configuration_utils import PretrainedConfig

class UMMMelDecoderConfig(PretrainedConfig):
    model_type = "UMMMelDecoder"

    def __init__(
        self,
        # mel decoder
        training_sample_rate=24000,
        latent_dim_in=32, 
        latent_dim_hz=25, 
        n_mel_bins_out=160,
        n_mel_channels_out=1, 
        num_conv_layers=4,
        conv_hidden_size=256,
        hidden_size=1024,
        return_complex=False,

        # losses
        w_loss_l1=1.0,
        w_loss_ssim=1.0,
        w_loss_adv=0.05,
        **kwargs
    ): 
        super().__init__(**kwargs)
        self.latent_dim_in = latent_dim_in
        self.latent_dim_hz = latent_dim_hz
        self.n_mel_bins_out = n_mel_bins_out
        self.n_mel_channels_out = n_mel_channels_out
        self.num_conv_layers = num_conv_layers
        self.conv_hidden_size = conv_hidden_size
        self.hidden_size = hidden_size
        self.return_complex = return_complex
        """
        @hanoihantrakul 18OCT2024 Implementation Notes
        The Mel-Decoder actually doesn't care what the sampling rate is. 
        This is because it receives as input a pre_vq_latent [batchsize, timesteps, latent_dims]
        and must output a mel spec of shape [batchsize, timesteps, n_mel_bins].

        This `n_mel_bins=160` could be representing a 24khz or a 44khz signal. It depends
        on how the lit_module loaded the data and computed the loss. 
        """
        self.training_sample_rate = training_sample_rate 

        self.w_loss_l1 = w_loss_l1
        self.w_loss_ssim = w_loss_ssim
        self.w_loss_adv = w_loss_adv

class UMMMelDecoder(nn.Module):
    """
    1NOV2024 @hanoihantrakul
    A Decoder which takes a latent (typically from Stage2 or Stage3 UMM) and transforms
    this into a mel spectrogram. The Mel spectrogram can be multi channel i.e.
    Mel Spec Mag-only Mono = 1 channel Mel

    25NOV2024 @hanoihantrakul
    This particular implementation only supports 1 channel Mel Spec 
    """

    def __init__(self, config: UMMMelDecoderConfig):
        super().__init__()
        # TODO: create a config object for this Decoder
        self.config = config
        self.proj_in = nn.Linear(config.latent_dim_in, config.hidden_size, bias=False)

        self.mel_decoder = ConvStacksWithDownUpSampling(
            config.conv_hidden_size, #config.conv_hidden_size
            config.hidden_size,  #config.hidden_size
            config.n_mel_bins_out, # assume 1 channel of mel spec (which could be mel-100, mel-120 or mel-160)
            num_layers=config.num_conv_layers, # default was 4
            downsampling=1,
            upsampling=4, # 25hz latent -> 100hz mel
        )

    def forward(self, x):
        """
        x: latent [batch_size, latent_dim_hz, latent_dim] 
        """
        # x = [batch_size, latent_timesteps, latent_dim] e.g. [4, 50, 32]
        x = self.proj_in(x)
        # x = [batch_size, latent_timesteps, hidden_size] e.g. [4, 50, 1024]
        x = self.mel_decoder(x)
        # x = [batch_size, mel_timesteps, n_mel_bins_out] e.g. [4, 200, 160]
        return x 
    
class UMMMelDecoderV2(nn.Module):
    """
    1NOV2024 @hanoihantrakul
    A Decoder which takes a latent (typically from Stage2 or Stage3 UMM) and transforms
    this into a mel spectrogram. The Mel spectrogram can be multi channel i.e.
    Mel Spec Mag-only Mono = 1 channel Mel
    Mel Spec Mag-only Stereo = 2 channel Mel
    Mel Spec Complex Stereo = 4 channel Mel

    To spike this approach, I am using the default `ConvStacksWithDownUpSampling` used as the 
    mel reconstruction head in traditional UMM Stage1-2-3 Training. 

    25NOV2024 @hanoihantrakul
    This implementation supports stereo mel spectrograms. I haven't implemented complex.


    Arguments:
        latent_dim_in: dim of latent (normally 32 if using Hanoi's ConformerUMM series)
        latent_dim_hz: latent hz (normally 25 if using Hanoi's ConformerUMM series)
        mel_c_out: number of output mel channels
        hidden_size: hidden size (int) of ResNet style blocks
        num_layers: number of layers of (int) of ResNet style block
    """

    def __init__(self, config: UMMMelDecoderConfig):
        super().__init__()
        # TODO: create a config object for this Decoder
        self.config = config
        self.proj_in = nn.Linear(config.latent_dim_in, config.hidden_size, bias=False)

        self.mel_decoder = ConvStacksWithDownUpSampling(
            config.conv_hidden_size, #config.conv_hidden_size
            config.hidden_size,  #config.hidden_size
            config.n_mel_bins_out * config.n_mel_channels_out, # the model will output the left and right mel channels together (bsz, time_steps, 2 * num_mels) which can then be tensor_split into the separate left and right mel specs.
            num_layers=config.num_conv_layers, # default was 4
            downsampling=1,
            upsampling=4, # 25hz latent -> 100hz mel
        )

    def forward(self, x):
        """
        x: latent [batch_size, latent_dim_hz, latent_dim] 
        """
        x = self.proj_in(x)
        x = self.mel_decoder(x)
        return x 
    

if __name__ == "__main__":
    inputs = {}
    BATCH_SIZE=1
    LATENT_HZ=25
    LATENT_TIMESTEPS=2*LATENT_HZ
    LATENT_DIM=32
    dummy_latent = torch.randn((BATCH_SIZE, LATENT_TIMESTEPS, LATENT_DIM)).to("cuda:0")

    model = UMMMelDecoder(latent_dim_in=LATENT_DIM, latent_dim_hz=LATENT_HZ).to("cuda:0")
    pred_mel = model(dummy_latent)
    print(pred_mel.shape)
        