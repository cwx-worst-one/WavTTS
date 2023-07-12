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

cfg = None

# normalize_audio_fp32 = NormalizeAudioToFloat32()
# normalize_audio = NormalizeAudio()

def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()


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
    text_lyrics: str,
    text_prompt: str,
    audio_prompt,
    # bs: int,
):
    if (audio_prompt is None and text_prompt is None):
        raise Exception('You must provide one of audio or text prompt. Please check examples at the bottom')
    if text_lyrics is None or len(text_lyrics) == 0:
        raise Exception('You must lyrics. Please check examples at the bottom')


    bs=3

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

    if audio_prompt_y is not None:
        audio_prompt_y = normalize_audio(audio_prompt_y)
    try:
        phoneme_ids = lyrics_tokenizer(text_lyrics) if text_lyrics else None
    except Exception as e:
        raise Exception(f"Could not convert lyrics to phonemes. Please try different lyrics {e}")

    item = {
        'lyrics': text_lyrics,
        'prompts': text_prompt,
        'lyrics_tokens': phoneme_ids,
        'audio.npy': audio_prompt_y,
        'vocals': None,
        'chroma': None
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

    if audio_prompt is not None:
        conditions = ["audio_prompt", "lyrics"]
    else:
        conditions = ["text_prompt", "lyrics"]

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

    sampled_audios = [torch_fp32_to_numpy_int16(wav) for wav in wavs]
    if audio_prompt_y is not None:
        gt_audio = (sample_rate, torch_fp32_to_numpy_int16(audio_prompt_y))
    else:
        gt_audio = None

    return (sample_rate, sampled_audios[0]), (sample_rate, sampled_audios[1]), (sample_rate, sampled_audios[2]), gt_audio, rtf


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

    cfg
    if cfg is None:
        cfg = load_config("recipes/l2v/conf/inference_mulan_phoneme.yaml", overrides)
    pl_module: ConditionalMulanPhonemeInferenceModule = cfg.pl_module
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
        value="justin bieber vocals pop song"
    )
    text_lyrics = gr.Textbox(
        label="Enter lyrics prompt for vocal generation",
        placeholder="Enter lyrics prompt",
        value="You just want attention, you don't want my heart"
    )
    audio_prompt = gr.Audio(
        label="Mulan Audio Prompt (Note: this replaces text prompt)"
    )
    # female singing pop ballad piano
    examples = [
        [
            "california dreaming",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/10120_full.mp3",
        ],
        [
            "near far wherever you are <n> I believe the heart does go on",
            # "/mnt/bn/audio-diffusion/data/karaoke_preview/full/30867_full.mp3", # rock - hard to hear
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/25336_full.mp3",
        ],
        [
            "you just want attention you don't want my heart",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/10096_full.mp3",
        ],
        [
            "We will be coming round the mountain <n> here she comes",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/21001_full.mp3",
         ],
        [
            "You can stand under my umbrella <n> ella ella eh eh",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/64494_full.mp3",
         ],
        [
            "I tried so hard to get so far <n> but in the end it doesn't even matter",
            None,
            "/mnt/bn/audio-diffusion/data/karaoke_preview/full/31030_full.mp3",
         ],
        [
            "I can show you the world, shining shimmering splendid",
            'justin bieber vocals pop song',
            None,
         ],
        [
            "you just want attention you don't want my heart",
            'upbeat hiphop female singing remix rihanna',
            None,
         ],
        [
            "You can stand under my umbrella <n> ella ella eh eh",
            'country music vocals guitar banjo taylor swift',
            None,
         ],
        [
            "Oh sometimes I got a good feeling. A feeling that I never ever had before",
            'avicii edm pop vocals singing sia',
            None,
         ],
    ]
    description = """
Demo for Lyrics to Vocal Music

* Mixture (use audio prompt) - audio_prompt,lyrics
* Mixture (use text prompt) - text_prompt,lyrics

Notes:
Text prompting is unstable - as we do not have a dedicated mulan model trained on vocals.
If the output vocals are silent, try adding "vocals", "singing", to the prompt.

Audio prompting is more stable, as long as you provide samples with singing
"""
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            text_lyrics,
            text_prompt,
            audio_prompt,
        ],
        outputs=[
            gr.Audio(label="Generated 1"),
            gr.Audio(label="Generated 2"),
            gr.Audio(label="Generated 3"),
            gr.Audio(label="Audio Ground truth"),
            gr.Textbox(label="RTF (3min avg)"),
        ],
        title="LyricsToVocalMusic",
        description=description,
        examples=examples
    )
    print('Ready to launch')
    demo.queue(concurrency_count=4)
    demo.launch(share=True)
