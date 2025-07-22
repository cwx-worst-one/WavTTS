import torch
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from recipes.mir2.utils.audio_utils import load_audio
import recipes.bigmusic.datasets.utils.zh_vocab_dev as zh_vocab_dev

def audio_to_torch(np_audio, hparams):
    audio_len_tokens = int(np.ceil(
        np_audio.shape[-1] / hparams['audio']['sampling_rate'] * 
        hparams['audio']['frame_rate']))
    chunk_len = int(hparams['audio']['sample_len'] * hparams['audio']['sampling_rate'])
    chunk_hop = int(hparams['audio']['hop_len'] * hparams['audio']['sampling_rate'])
    audio = torch.tensor(np_audio)
    audio = torch.nn.functional.pad(audio, (0, chunk_len))
    audio = audio.unfold(1, chunk_len, chunk_hop).squeeze()
    return audio, audio_len_tokens

def bigmusic_id2tag(tag_type, tag_id):
    #from IPython import embed; embed(using=False);
    meta_tag_type = zh_vocab_dev.tag_type_dict[tag_type]
    vocab2id = getattr(zh_vocab_dev, "VOCAB2ID_MIX_V4", None)
    for unified_tag, item in vocab2id.category_map[meta_tag_type].items():
        if item.value - 1 == tag_id:
            return unified_tag
    return 'none'

def prob_to_tag(tag_pred, tag_map):
    tag_map = { tag_type: { idx: category for category, idx in val.items()} for tag_type, val in tag_map.items() }
    res_tag = {}
    for tag_type in tag_map:
        prob = torch.sigmoid(tag_pred[tag_type]).detach().cpu().numpy()
        prob = np.mean(prob, axis=0)
        idx = np.argsort(-prob)
        if tag_type in ['ARTIST', 'MOOD', 'THEME', 'GENDER', 'TIMBRE']:
            res_tag[tag_type] = {bigmusic_id2tag(tag_type, i): round(prob[i], 4) for i in idx[:2]}
        elif tag_type == 'GENRE':
            res_tag[tag_type] = {bigmusic_id2tag(tag_type, i): round(prob[i], 4) for i in idx[:3]}
        elif tag_type == 'instruments':
            res_tag[tag_type] = {tag_map[tag_type][i]: round(prob[i], 4) for i in idx if prob[i]>0.099}
        elif tag_type == 'genres':
            res_tag[tag_type] = {tag_map[tag_type][i]: round(prob[i], 4) for i in idx[:3]}
        else:
            res_tag[tag_type] = {tag_map[tag_type][idx[0]]: round(prob[idx[0]], 4)}
    return res_tag


# load pretrained models
#yaml_path = "recipes/mir2/conf/karaoke/inference_vq_karaoke.yaml"
yaml_path = "recipes/umm2/conf/inference_stage4_token_bigmusic.yaml"
hparams = load_hyperpyyaml(open(yaml_path, "r", encoding="utf-8"))
model = hparams["model"]
required_modules = hparams["required_modules"]
tag2idx = hparams['mix_tag_types']
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
# split audio into a number of non-overlapping 50-second chunks

data = {'audio': audio.to(device)}
out = model.forward(data)

# post processing
out['vq_ids'] = out['vq_ids'].flatten().cpu().numpy()[:audio_len_tokens]
# each chunk has 50 * 25 = 1250 tokens, flatten into a single sequence, and cap at audio_len_tokens

pred_tags = prob_to_tag(out['tag_pred'], tag2idx)

# print results
print(out['vq_ids'].shape, out['vq_ids'])
open('test_vq_ids.txt', 'w').write('\n'.join([str(i) for i in out['vq_ids']]))
print([(g, out['tag_pred'][g].shape) for g in out['tag_pred']])
print(pred_tags)
