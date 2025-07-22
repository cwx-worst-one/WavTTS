import torch
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from recipes.mir2.utils.audio_utils import load_audio

def audio_to_torch(np_audio, hparams):
    audio_len_tokens = int(np.ceil(
        np_audio.shape[-1] / hparams['audio']['sampling_rate'] * 
        hparams['audio']['frame_rate']))
    chunk_len = int(hparams['audio']['max_duration'] * hparams['audio']['sampling_rate'])
    chunk_hop = int(hparams['audio']['sample_len'] * hparams['audio']['sampling_rate'])
    audio = torch.tensor(np_audio)
    audio = torch.nn.functional.pad(audio, (chunk_hop//2, chunk_len))
    audio = audio.unfold(1, chunk_len, chunk_hop).squeeze()
    return audio, audio_len_tokens


# load pretrained models
yaml_path = "recipes/umm2/conf/inference_stage3_wav2tokens.yaml"
hparams = load_hyperpyyaml(open(yaml_path, "r", encoding="utf-8"))
model = hparams["frontend"]
required_modules = hparams["required_modules"]
for module_name, loader_config in hparams["required_modules"].items():
    print(f"loading module {module_name}...")
    _args = {k: v for k, v in loader_config.items() if k != "loader"}
    loader = loader_config["loader"](**_args)
    model = loader.nn_load_model(model)

device = torch.device("cuda:7") 
model.to(device)

# run inference to extract tokens and tags
audio, _ = load_audio("test.mp3", sample_rate=hparams['audio']['sampling_rate'], downmix_to_mono=True)
audio, audio_len_tokens = audio_to_torch(audio, hparams)
data = {'audio': audio.to(device)}

out = model.forward(data)

#from IPython import embed; embed(using=False)

out['vq_ids'] = out['vq_ids'].flatten().cpu().numpy()[:audio_len_tokens]
open('test_vq_ids0.txt', 'w').write('\n'.join([str(i) for i in out['vq_ids']]))

print([(k, out[k].shape) for k in ['vq_ids', 'latent']])