import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class CAVPLoss(nn.Module):

    def __init__(self, intra_contrast_weight=1, clip_num=3):
        super().__init__()

        # Intra Contrast Weight:
        self.intra_contrast_weight = intra_contrast_weight
        self.clip_num = clip_num

    def forward(self, audio_embed, video_embed, logit_scale):

        device = audio_embed.device

        logits_per_audio = logit_scale * audio_embed @ video_embed.T
        logits_per_video = logit_scale * video_embed @ audio_embed.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        extra_loss = (F.cross_entropy(logits_per_audio, labels) +
                      F.cross_entropy(logits_per_video, labels)) / 2

        bs = logits_per_audio.shape[0]  # [b,b]
        assert bs % self.clip_num == 0
        s = (range(bs // self.clip_num), np.s_[:], range(bs // self.clip_num), np.s_[:])
        intra_logits_per_audio = logits_per_audio.reshape(
            bs // self.clip_num, self.clip_num, bs // self.clip_num,
            self.clip_num)[s]  # b' x clip_num x clip_num
        intra_logits_per_video = logits_per_video.reshape(
            bs // self.clip_num, self.clip_num, bs // self.clip_num,
            self.clip_num)[s]  # b' x clip_num x clip_num

        bs_intra, num_logits_intra, _ = intra_logits_per_audio.shape
        labels_intra = torch.arange(num_logits_intra, device=device,
                                    dtype=torch.long).unsqueeze(0).repeat(bs_intra, 1)
        # Intra Logits Per audio: bs x c
        intra_logits_per_audio = intra_logits_per_audio.reshape(bs_intra * num_logits_intra,
                                                                num_logits_intra)
        intra_logits_per_video = intra_logits_per_video.reshape(bs_intra * num_logits_intra,
                                                                num_logits_intra)
        labels_intra = labels_intra.reshape(bs_intra * num_logits_intra)
        # Intra Loss:
        intra_loss = (F.cross_entropy(intra_logits_per_audio, labels_intra) +
                      F.cross_entropy(intra_logits_per_video, labels_intra)) / 2

        total_loss = extra_loss + self.intra_contrast_weight * intra_loss

        # return {"contrastive_loss": total_loss} if output_dict else total_loss

        ret_dict = {"total_loss": total_loss, "extra_loss": extra_loss, "intra_loss": intra_loss}

        return ret_dict


if __name__ == "__main__":

    cavp_loss = CAVPLoss()

    audio_embed = torch.randn([30, 512])
    video_embed = torch.randn([30, 512])

    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    print(audio_embed.shape, video_embed.shape)

    logit_scale = torch.randn([1])

    extra_loss = cavp_loss(audio_embed, video_embed, logit_scale)

    print(extra_loss)
