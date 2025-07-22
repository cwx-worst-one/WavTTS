import torch
import torch.nn as nn
import numpy as np

class MultiFreqDiscriminator(nn.Module):
    def __init__(self, sr=24000, nch=1, channels=16):
        super().__init__()

        self.sr = sr
        self.nch = nch
        self.window = [64, 128, 256, 512, 1024, 2048, 4096]
        self.channels = channels
        self.eps = 1e-5

        self.discriminators = nn.ModuleList([FreqDiscriminator(2*nch, channels) for _ in range(len(self.window))])

    def forward(self, x, f_max=None):
        B, nch, _ = x.shape
        assert nch == self.nch

        if x.shape[-1] % self.sr != 0:
            x = nn.functional.pad(x, (0, self.sr - x.shape[-1] % self.sr), "constant", 0)
            
        # normalize power
        x_1s = x.float().reshape(B, nch, -1, self.sr)
        x_norm = x_1s / (x_1s.pow(2).sum((1,2,3), keepdim=True) + self.eps).sqrt()
        x_norm = x_norm.reshape(-1, self.sr)

        outputs = []
        feat_maps = []

        for i in range(len(self.discriminators)):
            x_spec = torch.stft(x_norm, self.window[i], self.window[i]//2, 
                                window=torch.hann_window(self.window[i]).to(x.device), 
                                return_complex=True)
            x_spec = x_spec.reshape(B, nch, -1, x_spec.shape[-1])
            x_RI = torch.cat([x_spec.real, x_spec.imag], dim=1).to(x)

            if f_max is not None:
                freq_dim = self.window[i] // 2 + 1
                if f_max < self.sr / 2:
                    valid_freq = int(np.ceil(freq_dim * 2 / self.sr * f_max))
                else:
                    valid_freq = freq_dim
                x_RI = x_RI[:,:,:valid_freq] + x_RI.sum() * 0.
            x_output, x_feat_map = self.discriminators[i](x_RI)
            outputs.append(x_output)
            feat_maps.append(x_feat_map)

        return outputs, feat_maps
        
class FreqDiscriminator(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super().__init__()

        eps = 1e-5
        self.discriminator = nn.ModuleList([])
        self.discriminator += [
            nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channel, hidden_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)), eps=eps),
                          nn.LeakyReLU(0.2, True)
                         ),
            nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(hidden_channel, hidden_channel*2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), eps=eps),
                          nn.LeakyReLU(0.2, True)
                         ),
            nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(hidden_channel*2, hidden_channel*4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)), eps=eps),
                          nn.LeakyReLU(0.2, True)
                         ),
            nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(hidden_channel*4, hidden_channel*8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), eps=eps),
                          nn.LeakyReLU(0.2, True)
                         ),
            nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(hidden_channel*8, hidden_channel*16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)), eps=eps),
                          nn.LeakyReLU(0.2, True)
                         ),
            nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(hidden_channel*16, hidden_channel*32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)), eps=eps),
                          nn.LeakyReLU(0.2, True)
                         ),
            nn.Conv2d(hidden_channel*32, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]

if __name__ == '__main__':
    discriminator = MultiFreqDiscriminator(sr=24000, channels=16)
    x = torch.randn(1, 1, 24000)

    if torch.cuda.is_available():
        discriminator = discriminator.cuda()
        x = x.cuda()

    s = 0
    for param in discriminator.parameters():
        s += np.product(param.shape)
    print('Model size: {} params'.format(s))

    outputs, feat_maps = discriminator(x, f_max=4000)
    print(len(outputs), len(feat_maps))
    print(feat_maps[0][0].shape, feat_maps[0][-1].shape, feat_maps[-1][0].shape, feat_maps[-1][-1].shape)
