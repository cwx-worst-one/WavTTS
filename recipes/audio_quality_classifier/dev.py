import torch
import glob
import soundfile as sf
import librosa
from collections import OrderedDict
from recipes.audio_quality_classifier.models.discriminator import MultiScaleSTFTDiscriminator



model = MultiScaleSTFTDiscriminator(
    filters=48
)
dummy_input = torch.randn(2, 1, 24000)


ckpt_path = "soundstream-step=592799-val_sdr=12.1354.ckpt"

state_dict = torch.load(ckpt_path)["state_dict"]
new_dict = OrderedDict()
for key in state_dict:
    if 'discriminator' in key:
        new_key = key.replace('discriminator.', '')
        new_dict[new_key] = state_dict[key]

model.load_state_dict(new_dict, strict=True)

target_folder = '../samantha/mcc40m_segments/'
files = sorted(glob.glob(f'{target_folder}/*.wav'))

out, _ = model(dummy_input)
for file in files:
    if '2519' in file:
    # audio, _ = sf.read(file)
        audio, _ = librosa.load(file, sr=24000)
        audio = torch.from_numpy(audio).float()[None, None, :]
        out, _ = model(audio)
        means = []
        for o in out:
            means.append(torch.mean(o))
        print(file, torch.mean(torch.stack(means)))
    