import os
import glob
from pathlib import Path
import torch
import torchaudio
import numpy as np
import soundfile as sf
from time import time
from recipes.diffusion.modules.pl_module import DiffusionModule
from recipes.diffusion.models.semantic_model.utils import (
    load_torch_script_module,
    w2v_bert_tokenization
)
from recipes.soundstream.models.vqgan import VQGAN_KL
from recipes.soundstream.modules.pl_module_vae import VocoderModule

os.makedirs('google_outs', exist_ok=True)

device = torch.device("cuda:0")
ckpt_path = 'diffusion-step=236799.ckpt'
semantic_model_path = '/opt/tiger/arnold_experiment/samantha/recipes/diffusion/assets/semantic.jit.pt'
semantic_centroid_path = '/opt/tiger/arnold_experiment/samantha/recipes/diffusion/assets/centroids_epoch_10.npy'
vocoder_model_path = '/opt/tiger/arnold_experiment/samantha/recipes/diffusion/assets/1000k_ckpt.pyt'

model = DiffusionModule.load_from_checkpoint(
    ckpt_path,
)
model.eval()
model.to(device)
model.sampler.set_device(device)

from recipes.diffusion.models.semantic_model.model import SSLFrontend
# ssl_frontend = SSLFrontend()
# semantic_model = load_torch_script_module(semantic_model_path, device)
# semantic_centers = torch.from_numpy(np.load(semantic_centroid_path)).float().to(device)
# ssl_frontend.to(device)
# semantic_model.to(device)
# ssl_frontend.eval()
# semantic_model.eval()


vocoder_model = VQGAN_KL(
        model_type='bytewave_wn',
        quant_token_dim=256,
        down_rates=[2, 3, 4, 4],
        upsample_rates=[4, 4, 3, 2],
        encoder_initial_channel=16,
        decoder_initial_channel=768,
        trunc_noise=False,
        smaller_encoder=True,
        init_cluster_size=32,
        dist=False,
)
def remove_ddp_module(ckpt):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for key in ckpt:
        new_key = key.replace('module.', '', 1)
        new_dict[new_key] = ckpt[key]
    return new_dict
ckpt = torch.load(vocoder_model_path, map_location='cpu')
state = remove_ddp_module(ckpt['G'])
vocoder_model.load_state_dict(state)
vocoder_model.to(device)
vocoder_model.eval()


# audio_path = '/opt/tiger/arnold_experiment/samantha/generated_samples/origin/1214647.wav'
# audio, sr = torchaudio.load(audio_path)

# context_token = w2v_bert_tokenization(
#     ssl_frontend, 
#     semantic_model, 
#     audio[None, 0, :].to(device), 
#     semantic_centers, 
#     device
# )

path_list = glob.glob('recipes/diffusion/assets/google_prompts/*/*.npy')

tokens_batch = []
for path in path_list:
    tokens = np.load(path)
    tokens_batch.append(torch.from_numpy(tokens))

tokens_batch = torch.stack(tokens_batch)

split_size = 20
tokens_batch = torch.split(tokens_batch, split_size, dim=0)

start_time = time()
with torch.no_grad():
    for i, batch in enumerate(tokens_batch):
        pred_emb = model.sampler(
            model=model.model,
            context=batch.to(device),
            num_items=batch.shape[0],
            num_chunks=4,
            num_steps=20,
            start=None,
            show_progress=True,
            angle_schedule='linear',
            classifier_free_guidance=2.5,
        ).detach()

        wavs_g = vocoder_model.decode(pred_emb.float()).detach()

        for wav_g, path in zip(wavs_g, path_list[i*split_size:(i+1)*split_size]):
            sf.write(
                f'google_outs/{str(Path(path).stem)}.wav',
                wav_g.cpu().numpy().T,
                24000,
            )

print(f'Inference RTF: {(time() - start_time)/(len(path_list)*10)}')