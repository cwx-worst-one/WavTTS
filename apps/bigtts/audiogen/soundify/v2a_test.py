import torch
import os
from torch import nn
import random
import numpy as np
from PIL import Image
from decord import VideoReader

from scipy.io.wavfile import write
from moviepy.editor import VideoFileClip, AudioFileClip

from apps.bigtts.audiogen.soundify.v2a.encoder import VideoEncoder
from apps.bigtts.audiogen.soundify.v2a.diffusion import Diffusion
from apps.bigtts.audiogen.soundify.v2a.vocoder import Vocoder


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)


def save_audio(audio, output_file, sr=32000):
    audio = audio * 32767
    audio = audio.astype("int16")
    write(output_file, sr, audio)
    return


def save_video(in_video, wave_path, out_path):

    video = VideoFileClip(in_video)
    video_dur = video.duration

    audio = AudioFileClip(wave_path)
    audio_dur = audio.duration

    min_duration = min(video_dur, audio_dur)

    print(video_dur, audio_dur, min_duration)

    if video_dur > min_duration:
        video = video.subclip(0, min_duration)
    else:
        audio = audio.subclip(0, min_duration)

    video_with_audio = video.set_audio(audio)

    video_with_audio.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
    print(out_path)


class Soundify_v2a(nn.Module):

    def __init__(self, cavp_ckpt, dit_ckpt, vocoder_ckpt):
        super().__init__()

        video_encoder = VideoEncoder()
        msg = video_encoder.load_state_dict(torch.load(cavp_ckpt))
        self.video_encoder = video_encoder.eval()
        print(f"loading video encoder {msg}")

        diffusion = Diffusion()
        msg = diffusion.load_state_dict(torch.load(dit_ckpt))
        self.diffusion = diffusion.eval()
        print(f"loading diffusion {msg}")

        vocoder = Vocoder()
        msg = vocoder.load_state_dict(torch.load(vocoder_ckpt))
        self.vocoder = vocoder.eval()
        print(f"loading vocoder {msg}")

    def read_video(self, in_video, target_fps=8, remove_caption=False):

        video_reader = VideoReader(in_video, num_threads=1)
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()

        target_num = int(vlen / float(fps) * float(target_fps))
        frame_indices = np.linspace(start=0, stop=vlen, num=target_num + 1)[0:-1].astype(int)

        out_frames = []
        for idx in frame_indices:
            frame = video_reader[idx].asnumpy()
            frame = Image.fromarray(frame).convert('RGB')
            if remove_caption:
                frame = frame.resize([280, 280])
                frame = frame.crop((28, 0, 252, 224))  # left, top, right, bottom
            else:
                frame = frame.resize([224, 224])
            frame = np.array(frame).astype('uint8')
            out_frames.append(frame)

        out_frames = np.asarray(out_frames)
        out_frames = torch.from_numpy(out_frames).to(dtype=torch.uint8)  # [t, h, w, c]
        out_frames = out_frames.permute(0, 3, 1, 2)  # [c, t, h, w]
        out_frames = (out_frames / 255.0).to(dtype=torch.float32)

        return out_frames  # [t, d, h, w]

    def fade_in(self, audio, start_point, end_point):

        fade_len = end_point - start_point
        fade_ratio = np.linspace(0.0, 1.0, fade_len)
        audio[:, start_point:end_point] = audio[:, start_point:end_point] * fade_ratio

        return audio

    def fade_out(self, audio, start_point, end_point):

        fade_len = end_point - start_point
        fade_ratio = np.linspace(1.0, 0.0, fade_len)
        audio[:, start_point:end_point] = audio[:, start_point:end_point] * fade_ratio

        return audio

    def set_silence(self, audio, start_point, end_point):

        audio[:, start_point:end_point] = 0.0

        return audio

    def post_process(self, wave):

        wave_len = wave.shape[1]
        wave = self.fade_in(wave, 0, 8000)
        wave = self.fade_out(wave, wave_len - 800, wave_len)

        return wave

    @torch.no_grad()
    def inference(self, frames, cfg_scale=7.5, step_num=50):

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            video_embs = self.video_encoder(frames)    
            latents = self.diffusion.ddim_sample(video_embs=video_embs,
                                                 step_num=step_num,
                                                 cfg_scale=cfg_scale)

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            wave = self.vocoder.decode(latents.float().permute(0, 2, 1))
            wave = wave.squeeze(1).cpu().numpy()
            wave = self.post_process(wave)

        return wave.squeeze(0)


if __name__ == "__main__":

    # in_video = ".deploy_cache/刹车1.mp4"
    in_video = ".deploy_cache/car.mp4"
    out_audio = ".deploy_cache/car.wav"
    out_video = ".deploy_cache/car_out.mp4"

    cavp_ckpt = ".deploy_cache/v2a_15w_32k_ft2_33000_encoder.ckpt"
    dit_ckpt = ".deploy_cache/v2a_15w_32k_ft2_33000_diffusion.ckpt"
    vocoder_ckpt = ".deploy_cache/v2a_15w_32k_ft2_33000_vocoder.ckpt"

    cfg_scale = 5.0
    step_num = 50

    set_seed(1234)
    device = "cuda"

    v2a_model = Soundify_v2a(cavp_ckpt=cavp_ckpt, dit_ckpt=dit_ckpt, vocoder_ckpt=vocoder_ckpt)
    v2a_model = v2a_model.to(device=device)

    with open("/mnt/bn/zxb-lq/workspace/samantha/apps/bigtts/audiogen/testdata/v2a.txt") as f:
        file_paths = f.read().splitlines()

    for in_video in file_paths:
        in_video = "/mnt/bn/zxb-lq/workspace/samantha/" + in_video
        file_name = in_video.split("/")[-1]

        out_audio = os.path.join(".deploy_cache", file_name.replace(".mp4", ".wav"))
        out_video = os.path.join(".deploy_cache", file_name.replace(".mp4", "_out.mp4"))
        # read video
        frames = v2a_model.read_video(in_video, target_fps=8, remove_caption=True)
        frames = frames.unsqueeze(0).to(device)
        # inference
        wave = v2a_model.inference(frames, cfg_scale = cfg_scale, step_num=step_num)
        save_audio(wave, out_audio)
        save_video(in_video, out_audio, out_video)
