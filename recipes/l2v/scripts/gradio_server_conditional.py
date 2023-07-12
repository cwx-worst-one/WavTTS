import os
import re
from time import perf_counter

import gradio as gr
import torch
from torchaudio import transforms

from recipes.musiclm.inference.utils import load_config
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import PhonemeTokenizer
from recipes.l2v.datasets.lyrics import LyricsCollator
from recipes.l2v.lightning.inference import ConditionalMulanPhonemeInferenceModule

arnold_task_id = 558104
batch_size = 1
sample_rate = 24000
n_audio_samples = 240000
device = "cuda"

# normalize_audio_fp32 = NormalizeAudioToFloat32()
# normalize_audio = NormalizeAudio()

def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()

from recipes.musiclm.datamodules.lit_datamodule import DataModule
from recipes.l2v.datasets.lyrics import LyricsCollator
from recipes.l2v.datasets.transforms.lyrics import get_chromagram

from torchaudio_augmentations import Compose

from recipes.musiclm.transforms.audio import (
    NormalizeAudioToFloat32,
    SetAudioDimensions,
    ToTensor,
    NormalizeAudio
)
normalize_audio = NormalizeAudio()
base_transforms = Compose(
    [
        ToTensor(), 
        SetAudioDimensions(), 
        NormalizeAudioToFloat32(),
    ]
)

def generate_audio(
    title: str,
    text_prompt: str,
    text_lyrics: str,
    audio_prompt,
    vocal_prompt,
    conditions: str,
    # bs: int,
):
    bs=1

    if audio_prompt is not None:
        audio_prompt_sr, audio_prompt_y = audio_prompt
        audio_prompt_y = base_transforms(audio_prompt_y.T)[0]
        if audio_prompt_sr != sample_rate:
            transform = transforms.Resample(audio_prompt_sr, sample_rate)
            audio_prompt_y = transform(audio_prompt_y)
        else:
            audio_prompt_y = audio_prompt_y
    else:
        audio_prompt_y = None

    if vocal_prompt is not None:
        vocal_prompt_sr, vocal_prompt_y = vocal_prompt
        vocal_prompt_y = base_transforms(vocal_prompt_y.T)[0]
        chroma = get_chromagram(vocal_prompt_y.numpy(), sr=vocal_prompt_sr, max_len=int(duration_sec*6))
        chroma = None
        if vocal_prompt_sr != sample_rate:
            transform = transforms.Resample(vocal_prompt_sr, sample_rate)
            vocal_prompt_y = transform(vocal_prompt_y)
        else:
            vocal_prompt_y = vocal_prompt_y
    else:
        vocal_prompt_y = None
        chroma = None

    if audio_prompt_y is not None:
        audio_prompt_y = normalize_audio(audio_prompt_y)
    if vocal_prompt_y is not None:
        vocal_prompt_y = normalize_audio(vocal_prompt_y, audio_prompt_y)
    
    phoneme_ids = lyrics_tokenizer(text_lyrics) if text_lyrics else None

    item = {
        'lyrics': text_lyrics,
        'prompts': text_prompt,
        'lyrics_tokens': phoneme_ids,
        'audio.npy': audio_prompt_y,
        'vocals': vocal_prompt_y,
        'chroma': chroma
    }
    batch = [item] * bs
    batch = lyrics_collator(batch)
    for k,t in batch.items():
        if torch.is_tensor(t): 
            batch[k] = t.cuda()

    # pl_module.extra_params.output_dir = str(base_output_path / output_folder_name)
    # pl_module.extra_params.conditions = conditions
    extra_params = pl_module.extra_params

    tik = perf_counter()

    conditions = conditions.split(',')
    coarse_samples = pl_module.coarse_module.predict(batch, extra_params, conditions)
    fine_samples = pl_module.fine_module.predict(coarse_samples, extra_params)
    
    coarse_samples = coarse_samples.view([bs, -1, extra_params.num_coarse])
    fine_samples = fine_samples.view([bs, -1, extra_params.num_fine])
    vqgan_inputs = (
        torch.cat([coarse_samples, fine_samples], dim=2)
        - torch.arange(extra_params.num_coarse + extra_params.num_fine, device=coarse_samples.device)
        * extra_params.soundstream_codebook_size
    )  # [b, t, n_codebook]
    vqgan_inputs = vqgan_inputs.transpose(
        1, 2
    )  # [b, t, n_codebook] -> [b, n_codebook, t]
    wavs = pl_module.requires["ss_dec"](vqgan_inputs).squeeze(1)


    tok = perf_counter()
    rtf = (tok - tik) / duration_sec

    sampled_audio = torch_fp32_to_numpy_int16(wavs[0])
    if audio_prompt_y is not None:
        gt_audio = (sample_rate, torch_fp32_to_numpy_int16(audio_prompt_y))
    else:
        gt_audio = None
    if vocal_prompt_y is not None:
        gt_vocal = (sample_rate, torch_fp32_to_numpy_int16(vocal_prompt_y))
    else:
        gt_vocal = None

    return (sample_rate, sampled_audio), gt_audio, gt_vocal, rtf


def ckpt_label_formatter(ckpt):
    trial_id = re.search(r"trials/(\d+?)/", ckpt).group(1)
    ckpt_fp = os.path.basename(ckpt)
    return f"{ckpt_fp} (Arnold trial {trial_id})"

if __name__ == "__main__":
    duration_sec = n_audio_samples / sample_rate
    overrides = { 
        'run_opts': { 'strategy': 'auto' }, 
        'predict_dataset': None,
        'pl_datamodule': None,
        'trainer': None,
        # 'pl_module': None # For fast load and debugging datasets
    }
    cfg = load_config("recipes/l2v/conf/inference_mulan_phoneme_conditional.yaml", overrides)
    pl_module: ConditionalMulanPhonemeInferenceModule = cfg.pl_module
    if pl_module is not None:
        pl_module.cuda()
        for k,v in pl_module.requires.items():
            if hasattr(v, 'cuda'):
                v.cuda()
    lyrics_collator: LyricsCollator = cfg.lyrics_collator
    lyrics_tokenizer: PhonemeTokenizer = cfg.lyrics_tokenizer

    title_prompt = gr.Textbox(
        label="Title",
    )

    text_prompt = gr.Textbox(
        label="Enter mulan 'text_prompt'. For accompaniment generation",
        placeholder="Enter mulan text prompt",
    )
    text_lyrics = gr.Textbox(
        label="Enter lyrics prompt for vocal generation",
        placeholder="Enter lyrics prompt",
        value="You just want attention, you don't want my heart"
    )
    audio_prompt = gr.Audio(
        label="Mulan Audio Prompt (Noote: this replaces text prompt)"
    )
    vocal_prompt = gr.Audio(label="Target Vocals")
    # vocal_prompt = gr.Audio(source="microphone", type="filepath", label="Target Vocals")
    print("Loading LyricsToVocal...")
    conditions_prompt = gr.Textbox(
        value="audio_prompt,lyrics", 
        label="Condition on: text_prompt|audio_prompt,lyrics,mulan_vocals,chroma")

        # file_ids_full = ['21423_0',
#  '28749_2',
#  '25336_1',
#  '29875_0',
#  '30867_0',
#  '10883_2',
#  '44410_2',
#  '64494_1']
# file_ids = ['10883_2', '44410_2', '64494_1'] # original ids
    # female singing pop ballad piano
    examples = [
        [
            "Mixture (audio prompt) + Conditions",
            None, "near far wherever you are <n> I believe the heart does go on",
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/30867_full.mp3",
            "/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/audio/29875_voca.mp3",
            "audio_prompt,lyrics,mulan_vocals,chroma"
         ],
        [
            "Mixture (audio prompt) + Conditions",
            None, "tick tock on the clock <n> don't stop make it pop",
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/10026_full.mp3",
            "/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/audio/10085_voca.mp3",
            "audio_prompt,lyrics,mulan_vocals,chroma"
         ],
        [
            "Mixture (audio prompt) + Conditions",
            None, "I tried so hard to get so far <n> but in the end it doesn't even matter",
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/10096_full.mp3",
            "/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/audio/10010_voca.mp3",
            "audio_prompt,lyrics,mulan_vocals,chroma"
         ],
        [
            "Vocal only + Conditions",
            None, "I can show you the world, shining shimmering splendid",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/audio/29875_voca.mp3",
            "lyrics,mulan_vocals,chroma"
         ],
        [
            "Mulan (text prompt) + Conditions",
            'upbeat edm female singing remix', "you just want attention you don't want my heart",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_vocal_accom/audio/10883_voca.mp3",
            "text_prompt,lyrics,mulan_vocals,chroma"
         ],
        [
            "Mixture (text prompt)",
            'country music vocals guitar banjo', "You can stand under my umbrella <n> ella ella eh eh",
            None,
            None,
            "text_prompt,lyrics",
         ],
        [
            "Mixture (audio prompt)",
            'country music vocals guitar banjo', "You can stand under my umbrella <n> ella ella eh eh",
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/44410_full.mp3",
            None,
            "text_prompt,lyrics",
         ],
        [
            "Instrumental only (text prompt)",
            'country music guitar banjo', None,
            None,
            None,
            "text_prompt",
         ],
    ]
    description = """
Demo for Lyrics to Vocal Music

Choose different conditioning types to generate different outputs. 
Click on an example at the very bottom for reference

* Mixture (use audio prompt) - audio_prompt,lyrics
* Mixture (use text prompt) - text_prompt,lyrics
* Vocals only - lyrics
* Instrumental - text_prompt
* Vocals + speaker condition - lyrics,mulan_vocals
* Vocals + speaker + melody condition - lyrics,mulan_vocals,chroma

Notes
- "chroma" = basic melody (6Hz)
- "mulan_vocals" = condition vocals on mulan audio tokens  
- Cannot be have both audio_prompt / text_prompt conditions at the same time.  
"""
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            title_prompt,
            text_prompt,
            text_lyrics,
            audio_prompt,
            vocal_prompt,
            conditions_prompt,
        ],
        outputs=[
            gr.Audio(label="Generated"),
            gr.Audio(label="Audio Ground truth"),
            gr.Audio(label="Speaker Ground truth"),
            gr.Textbox(label="RTF (3min avg)"),
        ],
        title="LyricsToVocalMusic",
        description=description,
        examples=examples
    )
    print('Ready to launch')
    demo.queue(concurrency_count=4)
    demo.launch(share=True)
