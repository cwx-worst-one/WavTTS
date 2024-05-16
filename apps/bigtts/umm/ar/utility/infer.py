import logging
import os
import struct

import librosa
import numpy as np
from scipy.io.wavfile import write

from samantha.utils.service.invoker import (
    alignment_invoke,
    invoke_asr,
    punctuation_recovery,
)

logging.basicConfig(level=logging.WARNING)

MAX_DUR = 20
MAX_SIL = 0.2
TRIM_SIL = 0.05
TAIL_THRESHOLD = 960  # 尾字停顿


def wav_to_audio_binary(data, fs):
    pcm_data = data.ravel().view("b").data

    file_size = 44 + len(pcm_data)
    header_data = b""
    header_data += b"RIFF"
    header_data += struct.pack("<I", file_size - 8)
    header_data += b"WAVE"
    header_data += b"fmt "

    format_tag = 0x0001
    channels = 1
    bit_depth = data.dtype.itemsize * 8
    bytes_per_second = fs * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)
    fmt_chunk_data = struct.pack(
        "<HHIIHH", format_tag, channels, fs, bytes_per_second, block_align, bit_depth
    )

    header_data += struct.pack("<I", len(fmt_chunk_data))  # 16
    header_data += fmt_chunk_data
    header_data += b"data"
    header_data += struct.pack("<I", data.nbytes)

    return header_data + pcm_data


def save_wav(audio, output_file, sr=24000):
    audio = audio * 32767
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return


def get_lignment(wav, sr, utt, lang):
    # 先识别
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)

    wav = (wav * 32768).astype(np.int16)
    audio_binary = wav_to_audio_binary(wav, 16000)
    text = invoke_asr(audio_binary, lang)
    # 对齐
    alignment_config = {
        "audio_info": {"sample_rate": 16000, "channel": 1, "format": "pcm"},
        "model": "big_model",
        "extra": {"text": "", "enable_phone": True},
    }
    alignment_config["extra"]["text"] = text
    res = alignment_invoke(utt, wav.tobytes(), alignment_config, "SKrwIGqYYx")
    # res = alignment_invoke(utt, wav.tobytes(), alignment_config, "oFDxHeMpJn")
    word_seqs = []
    for seg in res["results"]:
        for item in seg["alternatives"]:
            word_seqs.extend(item["words"])

    return text, word_seqs


def shift_word_seqs(word_seqs, shift_time):
    if shift_time == 0:
        return word_seqs
    for word in word_seqs:
        word["start_time"] = word["start_time"] - shift_time
        word["end_time"] = word["end_time"] - shift_time
    return word_seqs


# 加噪声
def process_internal_silence(wav, word_seqs):
    D = librosa.stft(y=wav, n_fft=2048, hop_length=300, win_length=1200)
    S = librosa.magphase(D)[0]
    energy = np.sqrt(np.sum(S**2, axis=0))
    pre_st, pre_et = word_seqs[0]["start_time"], word_seqs[0]["end_time"]
    sil_time_stap = []
    # 找到句中大于 0.5s 的长静音
    for word in word_seqs[1:]:
        st, et = word["start_time"], word["end_time"]
        if st - pre_et > MAX_SIL:
            sil_time_stap.append(((pre_et, st)))
        pre_st = st
        pre_et = et

    # 生成随机噪声
    signal_power = np.sum(wav**2)
    snr = 30  # 目标SNR
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.uniform(-1, 1, len(wav))  # 生成噪声
    noise = np.sqrt(noise_power) * noise / np.sqrt(np.sum(noise**2))  # 调整噪声能量

    wav_noisy = wav
    for sil_st, sil_et in sil_time_stap:
        st = int(sil_st * 24000)
        et = int(sil_et * 24000)
        # 长静音，并且能量低，需要加噪；
        if np.mean(energy[st // 300 : et // 300]) < 0.5:
            wav_noisy[st:et] = wav[st:et] + noise[st:et]
    return wav_noisy


# 加噪声低通滤波
def process_internal_silence_lowpass(wav, word_seqs):
    from scipy import signal

    D = librosa.stft(y=wav, n_fft=2048, hop_length=300, win_length=1200)
    S = librosa.magphase(D)[0]
    energy = np.sqrt(np.sum(S**2, axis=0))
    # pre_st, pre_et = word_seqs[0]["start_time"], word_seqs[0]["end_time"]
    pre_et = 0
    last_st = len(wav) / 24000.0
    sil_time_stap = []
    # 找到句中大于 MAX_SIL 的长静音
    for word in word_seqs:
        st, et = word["start_time"], word["end_time"]
        if st - pre_et > MAX_SIL:
            sil_time_stap.append(((pre_et, st)))
        pre_et = et
    if last_st - pre_et > MAX_SIL:
        sil_time_stap.append(((pre_et, last_st)))

    # 生成随机噪声
    signal_power = np.sum(wav**2)
    snr = 30  # 目标SNR
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.uniform(-1, 1, len(wav))  # 生成噪声
    b, a = signal.butter(4, 1.0 / 600, "lowpass")
    noise = signal.filtfilt(b, a, noise)  # 噪声过低通滤波，变为人耳范围之外
    noise = np.sqrt(noise_power) * noise / np.sqrt(np.sum(noise**2))  # 调整噪声能量

    wav_noisy = wav
    for sil_st, sil_et in sil_time_stap:
        st = int(sil_st * 24000)
        et = int(sil_et * 24000)
        # 长静音，并且能量低，需要加噪；
        if np.mean(energy[st // 300 : et // 300]) < 0.5:
            # print(noise[st:et])
            wav_noisy[st:et] = wav[st:et] + noise[st:et]
    return wav_noisy


# 截短长静音
def trim_internal_silence(wav, word_seqs):
    D = librosa.stft(y=wav, n_fft=2048, hop_length=300, win_length=1200)
    S = librosa.magphase(D)[0]
    energy = np.sqrt(np.sum(S**2, axis=0))
    pre_st, pre_et = word_seqs[0]["start_time"], word_seqs[0]["end_time"]
    sil_time_stap = []
    # 找到句中大于 0.5s 的长静音
    for word in word_seqs[1:]:
        st, et = word["start_time"], word["end_time"]
        if st - pre_et > 0.5:  # 截短0.5s以上静音
            sil_time_stap.append(((pre_et, st)))
        pre_st = st
        pre_et = et

    # 初始化一个空的音频列表，用于存储不包含在时间戳内的音频段
    wav_trim = []
    start_idx = 0
    for sil_st, sil_et in sil_time_stap:
        st = int((sil_st + 0.2) * 24000)
        et = int((sil_et - 0.2) * 24000)
        # 长静音，并且能量低，需要截短；
        if np.mean(energy[st // 300 : et // 300]) < 0.5:
            # 将不在时间戳范围内的音频段添加到音频列表中
            wav_trim.append(wav[start_idx:st])
            # 更新开始索引为静音结束位置
            start_idx = et
    # 将最后一段音频添加到音频列表中
    wav_trim.append(wav[start_idx:])
    # 将音频列表中的音频段连接起来
    wav_trim = np.concatenate(wav_trim)
    return wav_trim


# reverse trim
def reverse_trim(wav, word_seqs):
    # 找到句中大于 0.05s 的静音
    sil_time_stap = []
    pre_st, pre_et = word_seqs[0]["start_time"], word_seqs[0]["end_time"]
    for word in word_seqs[1:]:
        st, et = word["start_time"], word["end_time"]
        if st - pre_et > TRIM_SIL:
            sil_time_stap.append(((pre_et, st)))
        pre_st = st
        pre_et = et

    # 找到最长的静音
    if len(sil_time_stap) == 0:
        return wav
    sil_time_stap.sort(key=lambda x: x[1] - x[0], reverse=True)
    sil_st, sil_et = sil_time_stap[0]
    st = int(sil_st * 24000) + 100
    et = st + int(TRIM_SIL * 24000) + 100  # 0.05s
    pad_wav = wav[st:et]
    wav = np.concatenate((pad_wav, wav))
    return wav


# 截断长音频
def crop_wav(wav, word_seqs):
    valid_dur = word_seqs[-1]["end_time"] - word_seqs[0]["start_time"]

    # 判断尾部是否没截干净
    enough_tail_silence = (
        abs(wav.shape[0] - word_seqs[-1]["end_time"] * 24000) > TAIL_THRESHOLD
    )

    # 正常 < MAX_DUR and 尾部截取没问题
    if enough_tail_silence and valid_dur < MAX_DUR:
        text = " ".join([word["word"] for word in word_seqs])
        return wav, text, word_seqs
        # valid_st = int(word_seqs[0]["start_time"] * 24000)
        # valid_et = int(word_seqs[-1]["end_time"] * 24000)
        # word_seqs = shift_word_seqs(word_seqs, word_seqs[0]["start_time"])
        # return wav[valid_st: valid_et], text, word_seqs

    candidate_points = []
    sil_dur_thresh = [0.2, 0.1, 0.05, 0.02]  # (word, index, dur)
    min_prompt_dur = [5, 3, 3, 3]

    for thresh, min_dur in zip(sil_dur_thresh, min_prompt_dur):
        # import pdb; pdb.set_trace()
        text = [word_seqs[0]["word"]]
        pre_st, pre_et = word_seqs[0]["start_time"], word_seqs[0]["end_time"]
        sentence_st = pre_st
        res_et = 0
        res_word_index = 0
        i = 1
        for word in word_seqs[1:]:
            w = word["word"]
            st, et = word["start_time"], word["end_time"]
            text.append(w)
            if (
                st - pre_et > thresh and pre_et - sentence_st > min_dur
            ):  # 字间间隔超过 阈值 并且 长度超过设定的最短时长
                res_et = pre_et
                res_word_index = i - 1
                # candidate_points.append((word, res_word_index, res_et - sentence_st))
                # candidate_points.append((word, res_word_index, st - sentence_st))       # 放宽一点
                candidate_points.append(
                    (word, res_word_index, (pre_et + st) / 2 - sentence_st)
                )

            if et - sentence_st > MAX_DUR:
                break

            pre_st = st
            pre_et = et
            i = i + 1

        if len(candidate_points) > 0:
            word, res_word_index, res_et = candidate_points[-1]
            valid_st = int(sentence_st * 24000)
            valid_et = int(res_et * 24000)
            crop_wav = wav[valid_st : valid_st + valid_et]
            text = " ".join(text[: res_word_index + 1])
            word_seqs = shift_word_seqs(word_seqs, sentence_st)
            return crop_wav, text, word_seqs[: res_word_index + 1]

    print("processing prompt wav failed. recheck your prompt wav")

    return None


# 截断长音频
def crop_wav_right2left(wav, word_seqs):
    valid_dur = word_seqs[-1]["end_time"] - word_seqs[0]["start_time"]
    if valid_dur < MAX_DUR:
        text = " ".join([word["word"] for word in word_seqs])
        return wav, text, word_seqs

        # valid_st = int(word_seqs[0]["start_time"] * 24000)
        # valid_et = int(word_seqs[-1]["end_time"] * 24000)
        # word_seqs = shift_word_seqs(word_seqs, word_seqs[0]["start_time"])
        # return wav[valid_st: valid_et], text, word_seqs

    candidate_points = []
    sil_dur_thresh = [0.2, 0.1, 0.05, 0.02]  # (word, index, dur)
    min_prompt_dur = [5, 3, 3, 3]

    for thresh, min_dur in zip(sil_dur_thresh, min_prompt_dur):
        text = [word_seqs[-1]["word"]]
        pre_st, pre_et = word_seqs[-1]["start_time"], word_seqs[-1]["end_time"]
        sentence_et = pre_et
        res_st = 0
        res_word_index = 0
        i = 1

        for word in word_seqs[::-1][1:]:
            w = word["word"]
            st, et = word["start_time"], word["end_time"]
            text = [w] + text

            if pre_st - et > thresh and sentence_et - pre_st > min_dur:
                res_st = pre_st
                res_word_index = i - 1
                candidate_points.append((word, res_word_index, res_st))

            if sentence_et - st > MAX_DUR:
                break

            pre_st = st
            pre_et = et
            i = i + 1

        if len(candidate_points) > 0:
            word, res_word_index, res_st = candidate_points[-1]
            valid_st = int(res_st * 24000)
            valid_et = int(sentence_et * 24000)
            # crop_wav = wav[valid_st:valid_et]
            crop_wav = wav[valid_st:]  # 尾部不截取
            text = " ".join(text[-res_word_index - 1 :])
            word_seqs = shift_word_seqs(word_seqs, res_st)
            return crop_wav, text, word_seqs[-res_word_index - 1 :]

    print("processing prompt wav failed. recheck your prompt wav")

    return None


def write_word_seqs(word_seqs, outfile):
    with open(outfile, "w") as f:
        for word in word_seqs:
            w = word["word"]
            st = word["start_time"]
            et = word["end_time"]
            f.write(f"{w}\t{st}\t{et}\n")


import scipy.signal
import soundfile as sf


def process_prompt_wav(wavpath, out_wavpath, lang):
    utt = os.path.splitext(os.path.basename(wavpath))

    # wav, sr = librosa.load(wavpath, sr=None)
    wav, sr = sf.read(wavpath)
    if len(wav.shape) > 1:
        wav = wav[:, 0]

    print(f"processing {wavpath}")
    # 对齐
    text, word_seqs = get_lignment(wav, sr, utt, lang)

    # 处理采样率，非24k音频重采样到24k.
    if sr != 24000:
        # wav_24k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24000)
        wav_24k = scipy.signal.resample(wav, int(len(wav) * 24000 / sr))
    else:
        wav_24k = wav

    write_word_seqs(word_seqs, out_wavpath.replace(".wav", ".txt"))
    res = crop_wav(wav_24k, word_seqs)
    # res = crop_wav_right2left(wav_24k, word_seqs)
    if res is None:
        return ""
    else:
        wav_cropped, text, word_seqs_croped = res

    # 截短长静音
    # processed_wav = trim_internal_silence(wav_cropped, word_seqs_croped)

    # 处理句中的长静音
    # processed_wav = process_internal_silence(wav_cropped, word_seqs_croped)
    # processed_wav = process_internal_silence_lowpass(wav_cropped, word_seqs_croped)
    processed_wav = wav_cropped

    # 补帧
    # processed_wav = reverse_trim(processed_wav, word_seqs_croped)

    save_wav(processed_wav, out_wavpath, sr=24000)

    # 标点恢复
    with open(out_wavpath, "rb") as buffer:
        result = punctuation_recovery(buffer.read(), "oFDxHeMpJn")
        if result is not None:
            text = result["text"]
            print(text)
        else:
            print("punctuation_recovery fail!")

    return text


if __name__ == "__main__":
    # indir = "/mnt/bn/jcong5/workspace5/temperture/samantha-preprocess/preprcess_prompt_wav_testdata/raw"
    # outdir = "/mnt/bn/jcong5/workspace5/temperture/samantha-preprocess/preprcess_prompt_wav_testdata/processed"
    indir = "/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_demo_9.0"
    outdir = "/mnt/bn/cjw-lq-1/project/kainan/test/bigtts_testset/icl_testset_demo_9.0/processed"

    # for lang in os.listdir(indir):
    for lang in ["zh"]:
        lang_dir = os.path.join(indir, lang)
        out_lang_dir = os.path.join(outdir, lang)

        os.makedirs(out_lang_dir, exist_ok=True)

        fout = open(os.path.join(out_lang_dir, "meta.txt"), "w")

        for item in os.listdir(lang_dir):
            wavepath = os.path.join(lang_dir, item)
            out_wavepath = os.path.join(out_lang_dir, item)
            text = process_prompt_wav(wavepath, out_wavepath, lang)
            if text == "":
                continue
            fout.write(f"{text}|{out_wavepath}\n")
            fout.flush()
