import os
import sys


sys.path.append(os.getcwd())

import thop
import torch

from wavtts.model import CFM, DiT


""" ~155M """
# transformer =       DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2)
# transformer =       DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2, text_dim = 512, conv_layers = 4)
# transformer =       DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2, text_dim = 512, conv_layers = 4, long_skip_connection = True)
# transformer =     MMDiT(dim = 512, depth = 16, heads = 16, ff_mult = 2)

""" ~335M """
# FLOPs: 622.1 G, Params: 333.2 M
# FLOPs: 363.4 G, Params: 335.8 M
transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


model = CFM(transformer=transformer)
target_sample_rate = 16000
wav_frame_len = 160
duration = 20
frame_length = int(duration * target_sample_rate / wav_frame_len)
text_length = 150

flops, params = thop.profile(
    model, inputs=(torch.randn(1, frame_length, wav_frame_len), torch.zeros(1, text_length, dtype=torch.long))
)
print(f"FLOPs: {flops / 1e9} G")
print(f"Params: {params / 1e6} M")
