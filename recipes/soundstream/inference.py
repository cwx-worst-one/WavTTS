import torch
import torchaudio
import soundfile as sf
from recipes.soundstream.models.vqgan import VQGAN
from recipes.soundstream.models.transformer import TransformerRVQ
from recipes.soundstream.modules.pl_module import SoundstreamModule
from recipes.soundstream.models.modules.ms_tfd import MultiScaleSTFTDiscriminator

from recipes.soundstream.utils.audio_utils import (
    pad_audio, 
    enframe, 
    deframe
)

device = torch.device("cuda:0")
ckpt_path = 'logs/soundstream/48k/checkpoints/soundstream-step=015649-val_sdr=7.7086.ckpt'
# ckpt_path = 'soundstream-step=214000.ckpt'

torch.manual_seed(0)

# gen = VQGAN(
#     model_type='bytewave_wn_causal',
#     num_res=12,
#     quant_token_num=1024,
#     quant_token_dim=256,
#     quant_beta=0.25,
#     down_rates=[2, 2, 10, 12],
#     upsample_rates=[10, 6, 4, 2],
#     encoder_initial_channel=16,
#     decoder_initial_channel=768,
#     trunc_noise=False,
#     smaller_encoder=True,
#     init_cluster_size=32,
#     dist=False,
# )

gen = TransformerRVQ(
    num_res=12,
    quant_token_num=1024,
    quant_token_dim=256,
    quant_beta=0.25,
    init_cluster_size=32,
    window_size=64,
    hop_size=32,
    encoder_dim=256,
    encoder_heads=8,
    encoder_resample_rates=[2, 2, 2, 2, 2],
    encoder_resample_interval=3,
    decoder_dim=256,
    decoder_heads=8,
    decoder_resample_rates=[2, 2, 2, 2, 2],
    decoder_resample_interval=3,
    attn_dropout=0.1,
    ff_dropout=0.1,
    use_checkpoint=False,
    dist=False,
)

dis = MultiScaleSTFTDiscriminator(
    filters=32,
    pretrain=False
)


model = SoundstreamModule.load_from_checkpoint(ckpt_path, generator=gen, discriminator=dis, balancer=None)
model.to(device)
model.eval()

test_input, _ = torchaudio.load("/mnt/bd/sami-weitsung-2/mss_datasets/mtg_jamendo_raw/00/1002000.mp3")
test_input = torch.mean(test_input, dim=0, keepdim=True)

input_chunk_samples = 48000
hop_samples = 48000 
batch_size = 20

origin_len = test_input.shape[-1]
with torch.no_grad():
    # Pad
    test_input = pad_audio(
        test_input, 
        segment_samples=input_chunk_samples,
        hop_samples=hop_samples,
    )
    # Enframe
    test_input = enframe(
        test_input, 
        segment_samples=input_chunk_samples,
        hop_samples=hop_samples,
    )
   
    _dict = {kk :{k : False for k in range(1024)} for kk in range(12)}
    # Inference
    out = []
    for b in torch.split(test_input, batch_size):
        o, _, quant_index, _ = model.generator(b.to(device), warmup=False)
        out.append(o.detach())
        
        for idx, batch_code in enumerate(quant_index):
            for code in batch_code:
                for c in code:
                    _dict[idx][c.item()] = True
    
    for k in _dict:
        cnt = 0
        for kk in _dict[k]:
            if _dict[k][kk] is True:
                cnt += 1
        print(f"Layer {k} use {cnt/1024} unique codes")
    # Deframe and depad
    out = deframe(torch.cat(out, dim=0), hop_samples=hop_samples)[..., :origin_len]

sf.write("recons_test.wav", out[0].detach().cpu().numpy().T, 48000)

