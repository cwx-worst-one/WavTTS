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
    audio = torch.nn.functional.pad(audio, ((chunk_len-chunk_hop)//2, chunk_len))
    audio = audio.unfold(1, chunk_len, chunk_hop).squeeze()
    return audio, audio_len_tokens


def prob_to_tag(tag_pred, tag_map):
    tag_map = { tag_type: { idx: category for category, idx in val.items()} for tag_type, val in tag_map.items() }
    res_tag = {}
    for tag_type in tag_map:
        prob = torch.sigmoid(tag_pred[tag_type]).detach().cpu().numpy()
        prob = np.mean(prob, axis=0)
        idx = np.argsort(-prob)
        if tag_type == 'instruments':
            res_tag[tag_type] = {tag_map[tag_type][i]: round(prob[i], 4) for i in idx if prob[i]>0.099}
        elif tag_type == 'genres':
            res_tag[tag_type] = {tag_map[tag_type][i]: round(prob[i], 4) for i in idx[:3]}
        else:
            res_tag[tag_type] = {tag_map[tag_type][idx[0]]: round(prob[idx[0]], 4)}
    return res_tag


# load pretrained models
#yaml_path = "recipes/mir2/conf/karaoke/inference_vq_karaoke.yaml"
yaml_path = "recipes/mir2/conf/karaoke/inference_stage3_karaoke.yaml"
hparams = load_hyperpyyaml(open(yaml_path, "r", encoding="utf-8"))
model = hparams["model"]
required_modules = hparams["required_modules"]
tag2idx = hparams['extra_params']['tag_map']
for module_name, loader_config in hparams["required_modules"].items():
    print(f"loading module {module_name}...")
    _args = {k: v for k, v in loader_config.items() if k != "loader"}
    loader = loader_config["loader"](**_args)
    model = loader.nn_load_model(model)

device = torch.device("cuda") 
model.to(device)


# run inference to extract tokens and tags
audio, _ = load_audio("test.mp3", sample_rate=hparams['audio']['sampling_rate'], downmix_to_mono=True)
audio, audio_len_tokens = audio_to_torch(audio, hparams)
audio = audio.unsqueeze(0)
audio = audio.to(device)

########### load it with simple inference ################

data = {'audio': audio}
model = hparams["model"]
required_modules = hparams["required_modules"]
tag2idx = hparams['extra_params']['tag_map']
for module_name, loader_config in hparams["required_modules"].items():
    print(f"loading module {module_name}...")
    _args = {k: v for k, v in loader_config.items() if k != "loader" and k != "initializer"}
    loader = loader_config["loader"](**_args)
    model = loader.nn_load_model(model)
model.to(device)
out = model.forward(data)
from IPython import embed; embed(using=False); 

breakpoint()
# post processing
out['vq_ids'] = out['vq_ids'].flatten().cpu().numpy()[:audio_len_tokens]
pred_tags = prob_to_tag(out['tag_pred'], tag2idx)

# post processing
out['vq_ids'] = out['vq_ids'].flatten().cpu().numpy()[:audio_len_tokens]
pred_tags = prob_to_tag(out['tag_pred'], tag2idx)


########### load it the same way as AR training ################

loader_config = next(iter(hparams["required_modules"].values()))
breakpoint()
initializer = loader_config["initializer"]
hpath = loader_config["ckpt_path"]
requires = initializer(hpath, local_rank=0, cache_dir='.module_cache/bigmusic')
model = requires['UMM2_30s']
out = model.wav2token(audio.to(device))
print(out)
