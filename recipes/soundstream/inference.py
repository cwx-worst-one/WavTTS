import torch
from recipes.diffusion.models.vocoder_model.utils import init_vocoder
from glob import glob
import soundfile as sf
model = init_vocoder(
    'soundstream-step=890000-val_sdr=11.9869-EMA.ckpt',
    0,
    None,
    adapt_hopper=True,
)['vocoder']
# put audio here
# input_audio = torch.randn(1, 1, 24000*10).cuda()
paths = glob('unified_val/music/*.wav')

for path in paths:
    output_path = path[:-4] + "_recons.wav"

    audio, sr = sf.read(path)
    input_audio = torch.from_numpy(audio.T[None, None]).float().to('cuda:0')
    with torch.no_grad():
        output_audio = model(input_audio)
        encoder_out = model.encode(input_audio)
        sample, kl_loss, std_mean = model.sample(encoder_out, deterministic=False)
        decoder_out = model.decode(sample)
    print(decoder_out.shape)

    sf.write(output_path, decoder_out.cpu().detach().numpy()[0].T, samplerate=24000)