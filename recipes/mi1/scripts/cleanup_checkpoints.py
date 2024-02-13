import torch
from glob import glob
from recipes.mi1.models.music_sft import MI1_MusicClassificationMusicSFT


for fp in glob("m1/*.ckpt"):
    ckpt = torch.load(fp)
    state_dict = {}
    for k in ckpt["state_dict"]:
        if "audio_tokenizer" not in k:
            state_dict[k] = ckpt["state_dict"][k]
   
    ckpt["state_dict"] = state_dict

    torch.save(ckpt, fp)
    model = MI1_MusicClassificationMusicSFT.load_from_checkpoint(fp)
    print(model.tag_tokenizer.vocab)