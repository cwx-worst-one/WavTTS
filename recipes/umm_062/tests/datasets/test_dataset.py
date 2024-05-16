

# from hyperpyyaml import load_hyperpyyaml
# print("load_hyperpyyaml")
# cfg = load_hyperpyyaml(open("recipes/umm/conf/stage1_music.yaml", "r", encoding="utf8"))
# # print(f"cfg={cfg}")
# dm = cfg["pl_datamodule"]
# # print(dm)

import time

import numpy
import torch
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity, profile, record_function

from recipes.datasets.mcc.mix import (
    MCCInstrumentalDataset,
    MCCVocalDataset,
    MixWebDataModule,
)
from recipes.musiclm.transforms.audio import FastNormalizeAudio, NormalizeAudio


def set_random_seed(seed=123):
    """Set random seed manully to get deterministic results"""
    import random
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def test_MixWebDataModule():
    set_random_seed()

    batch_size = 8
    shuffle_buffer_size = 10
    num_workers = 2
    sample_rate = 24000
    loop_num = 100

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        pl_datamodule = MixWebDataModule(
            sample_rate=sample_rate,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_workers=num_workers,
        )

        train_loader = pl_datamodule.train_dataloader()

        for i, batch in enumerate(train_loader):
            print(f"======== {i}/{loop_num}")
            text = batch["text"]
            audio = batch["audio"]
            print(f"text={text} audio={audio.shape}")
            if i >= loop_num:
                break
    # prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10, max_src_column_width=1000, max_name_column_width=1000))


def test_NormalizeAudio():
    audio_length = 5e6
    loop_num = 500
    audio = torch.rand(size=[1, int(audio_length)])
    ref_model = NormalizeAudio()
    opt_model = FastNormalizeAudio()

    ref_output = ref_model(audio)
    opt_output = opt_model(audio)

    torch.allclose(ref_output, opt_output)

    start_time = time.perf_counter()
    for _ in range(loop_num): 
        output = ref_model(audio)
    ref_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    for _ in range(loop_num): 
        output = opt_model(audio)
    opt_time = time.perf_counter() - start_time
    print(f"ref_time={ref_time/loop_num} opt_time={opt_time/loop_num}")

