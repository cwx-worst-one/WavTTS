# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys

from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from pydub import AudioSegment, silence
from transformers import pipeline

from wavtts.model import CFM
from wavtts.model.utils import convert_char_to_pinyin, get_tokenizer


_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

# -----------------------------------------

target_sample_rate = 16000  # WavTTS waveform default
use_bfloat16 = True  # True, False
wav_frame_len = 160
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"  # fixed WavTTS inference ODE method
nfe_step = 50
cfg_strength = 3.0
sway_sampling_coef = -1.0
timestep_mapping = "power"
timestep_power = 2.0
shift = 3.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks



# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
    cfm_kwargs=None,
    waveform_kwargs=None,
):
    if vocab_file == "":
        vocab_file = str(files("wavtts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    if waveform_kwargs is None:
        waveform_kwargs = dict(
            wav_frame_len=wav_frame_len,
            target_sample_rate=target_sample_rate,
        )

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(
            **model_cfg,
            text_num_embeds=vocab_size,
            wav_frame_len=waveform_kwargs.get("wav_frame_len", wav_frame_len),
        ),
        waveform_kwargs=waveform_kwargs,
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
        **cfm_kwargs,
    ).to(device)

    # Keep inference-time audio geometry available even when CFM is wav-only and has no MelSpec module.
    model.target_sample_rate = int(waveform_kwargs.get("target_sample_rate", target_sample_rate))

    model = load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name

        aseg = AudioSegment.from_file(ref_audio_orig)

        # 1. try to find long silence for clipping
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (1)")
                break
            non_silent_wave += non_silent_seg

        # 2. try to find short silence for clipping if 1. failed
        if len(non_silent_wave) > 12000:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (2)")
                    break
                non_silent_wave += non_silent_seg

        aseg = non_silent_wave

        # 3. if no proper silence found for clipping
        if len(aseg) > 12000:
            aseg = aseg[:12000]
            show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(temp_path, format="wav")
        ref_audio = temp_path

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    timestep_mapping=timestep_mapping,
    timestep_power=timestep_power,
    shift=shift,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            timestep_mapping=timestep_mapping,
            timestep_power=timestep_power,
            shift=shift,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


# infer batches


def _model_audio_sample_rate(model_obj):
    if hasattr(model_obj, "target_sample_rate"):
        return int(model_obj.target_sample_rate)
    return int(target_sample_rate)


def _supports_cuda_autocast(device):
    return use_bfloat16 and device is not None and "cuda" in str(device) and torch.cuda.is_available()


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=50,
    cfg_strength=3.0,
    sway_sampling_coef=-1,
    timestep_mapping="power",
    timestep_power=2.0,
    shift=3.0,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    model_sample_rate = _model_audio_sample_rate(model_obj)

    rms = torch.sqrt(torch.mean(torch.square(audio))).clamp_min(1e-8)
    if rms < target_rms:
        audio = audio * target_rms / rms

    if sr != model_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, model_sample_rate)
        audio = resampler(audio)

    audio = audio.to(device)

    frame_len = int(getattr(model_obj, "wav_frame_len", getattr(model_obj, "num_channels", 160)))
    ref_len_samples = audio.shape[-1]
    ref_full_frames = ref_len_samples // frame_len
    ref_len_aligned = max(ref_full_frames * frame_len, frame_len)
    if ref_len_aligned > ref_len_samples:
        audio = torch.nn.functional.pad(audio, (0, ref_len_aligned - ref_len_samples), value=0.0)
    else:
        audio = audio[..., :ref_len_aligned]
    ref_audio_len = ref_len_aligned

    generated_waves = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        if fix_duration is not None:
            duration = int(fix_duration * model_sample_rate)  # samples
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # inference
        with torch.inference_mode():
            if _supports_cuda_autocast(device):
                audio_bf16 = audio.to(torch.bfloat16)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated, _ = model_obj.sample(
                        cond=audio_bf16,
                        text=final_text_list,
                        duration=duration,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        timestep_mapping=timestep_mapping,
                        timestep_power=timestep_power,
                        shift=shift,
                        use_epss=timestep_mapping == "sway_sampling",
                    )
            else:
                generated, _ = model_obj.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    timestep_mapping=timestep_mapping,
                    timestep_power=timestep_power,
                    shift=shift,
                    use_epss=timestep_mapping == "sway_sampling",
                )
            del _

            generated = generated.to(torch.float32)  # [1, N_total]
            cut = ref_audio_len
            generated_wave = generated[0, cut:].contiguous()

            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            generated_wave = generated_wave.cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], model_sample_rate
            else:
                yield generated_wave, None

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, _ = next(result)
                    generated_waves.append(generated_wave)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * model_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            yield final_wave, model_sample_rate, None

        else:
            yield None, model_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
