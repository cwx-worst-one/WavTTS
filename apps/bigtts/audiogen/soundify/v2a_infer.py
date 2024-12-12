import torch
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



class Soundify_video:

    def __init__(self, target_fps=8, mean_threshold=10, std_threshold=10, remove_caption=True):

        self.target_fps = target_fps
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.remove_caption = remove_caption

    def detect_black(self, mini_frames):

        gray = 0.299 * mini_frames[..., 0] + 0.587 * mini_frames[..., 1] + 0.114 * mini_frames[...,
                                                                                               2]
        gray_mean = torch.mean(gray, axis=[0, 2])
        gray_std = torch.std(gray, axis=[0, 2])

        cnt = 0
        for top in range(40, -1, -1):
            if gray_mean[top] < self.mean_threshold and gray_std[top] < self.std_threshold:
                cnt += 1
                if cnt == 3:
                    break
        top = top + cnt

        cnt = 0
        for bottom in range(60, 100, 1):
            if gray_mean[bottom] < self.mean_threshold and gray_std[bottom] < self.std_threshold:
                cnt += 1
                if cnt == 3:
                    break
        bottom = bottom - cnt

        return top, bottom

    def resize(self, frame, top, bottom):

        w, h = frame.size
        ratio = w / h

        if ratio < 9 / 14:
            top = np.clip(top, 10, 40)
            bottom = np.clip(bottom, 60, 90)

        if self.remove_caption:
            w_ratio = 0.85
        else:
            w_ratio = 1.0

        img_top = int(top / 100 * h)
        img_bottom = int(bottom / 100 * h)
        new_h = int((img_bottom - img_top) * w_ratio)
        img_bottom = img_top + new_h

        new_w = np.clip(int(new_h * ratio), w * 0.85, w)

        img_left = (w - new_w) // 2
        img_right = img_left + new_w

        frame = frame.crop((img_left, img_top, img_right, img_bottom))
        frame = frame.resize([224, 224])

        return frame

    def load(self, in_video):

        video_reader = VideoReader(in_video, num_threads=1)
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()

        target_num = int(vlen / float(fps) * float(self.target_fps))
        frame_indices = np.linspace(start=0, stop=vlen, num=target_num + 1)[0:-1].astype(int)

        mini_frames = []
        for idx in frame_indices:
            frame = video_reader[idx].asnumpy()
            frame = Image.fromarray(frame).convert('RGB')
            w, h = frame.size
            frame = frame.resize([int(100 * w / h), 100])
            frame = np.array(frame).astype('uint8')
            mini_frames.append(frame)
        mini_frames = torch.from_numpy(np.asarray(mini_frames))

        top, bottom = self.detect_black(mini_frames)

        out_frames = []
        for idx in frame_indices:
            frame = video_reader[idx].asnumpy()
            frame = Image.fromarray(frame).convert('RGB')
            frame = self.resize(frame, top, bottom)
            frame = np.array(frame).astype('uint8')
            out_frames.append(frame)

        out_frames = np.asarray(out_frames)
        out_frames = torch.from_numpy(out_frames).to(dtype=torch.uint8)  # [t, h, w, c]
        out_frames = out_frames.permute(0, 3, 1, 2)  # [c, t, h, w]
        out_frames = (out_frames / 255.0).to(dtype=torch.float32)

        return out_frames


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

            # remove caption
            w, h = frame.size
            if remove_caption and w > h:
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

    def post_process(self, wave):

        wave_len = wave.shape[1]
        wave = self.fade_in(wave, 0, 32000)
        wave = self.fade_out(wave, wave_len - 32000, wave_len)

        return wave

    @torch.no_grad()
    def inference(self, frames, cfg_scale=7.5, step_num=50):

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            video_embs = self.video_encoder.infer(frames)
            latents = self.diffusion.ddim_sample(video_embs=video_embs,
                                                 step_num=step_num,
                                                 cfg_scale=cfg_scale)

        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            wave = self.vocoder.decode(latents.float().permute(0, 2, 1))
            wave = wave.squeeze(1).cpu().numpy()
            wave = self.post_process(wave)

        return wave.squeeze(0)


if __name__ == "__main__":

    in_video = ".deploy_cache/case3.mp4"

    cavp_ckpt = ".deploy_cache/v2a_0.7b_0.3_v2_20k_encoder.ckpt"
    dit_ckpt = ".deploy_cache/v2a_0.7b_0.3_v2_20k_diffusion.ckpt"
    vocoder_ckpt = ".deploy_cache/v2a_0.7b_0.3_v2_20k_vocoder.ckpt"

    cfg_scale = 4.5
    step_num = 50

    device = "cuda"

    v2a_model = Soundify_v2a(cavp_ckpt=cavp_ckpt, dit_ckpt=dit_ckpt, vocoder_ckpt=vocoder_ckpt)
    v2a_model = v2a_model.to(device=device)

    v2_reader = Soundify_video(target_fps=8, mean_threshold=10, std_threshold=10, remove_caption=True)

    set_seed(1234)




    # read video
    frames = v2_reader.load(in_video)
    frames = frames.unsqueeze(0).to(device)

    out_audio = ".deploy_cache/case3.wav"
    out_video = ".deploy_cache/case3.mp4"
    # inference
    wave = v2a_model.inference(frames, cfg_scale=cfg_scale, step_num=step_num)
    save_audio(wave, out_audio)
    save_video(in_video, out_audio, out_video)
