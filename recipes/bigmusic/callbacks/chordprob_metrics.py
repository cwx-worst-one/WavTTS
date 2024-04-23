import os
from pathlib import Path
import json
import glob
import numpy as np
import torch
import pytorch_lightning as pl
from recipes.musiclm.inference.utils import load_wav
from recipes.bigmusic.utils.format_utils import update_json
from torchaudio.functional import resample

from recipes.chord.requires.model_initializer import init_chord

class ChordProbMetricsCallback(pl.Callback):
    def __init__(self, chord_model_path=None):
        super().__init__()
        self.chord_model_path = chord_model_path

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        # pl_module.requires.keys(): ['diffusion', 'sampler', 'vocoder', 'reranker', 'chord']
        sample_rate = pl_module.extra_params.sample_rate
        chordprob_callback = pl_module.extra_params.get("chordprob_callback", False)
        if not chordprob_callback:
            return
        
        chord_model = pl_module.requires["chord"]
        if 'output_paths' in pl_module.extra_params:
            generated_output_fps = pl_module.extra_params.output_paths
        else:
            output_dir = pl_module.extra_params.output_dir
            generated_output_fps = list(Path(output_dir).glob('**/*.generated.wav'))
        
        run_chordprob_metrics(chord_model, generated_output_fps, device=pl_module.device, sample_rate=sample_rate)


def run_one_sample(chord_model, generated_output_fp, device='cuda', sample_rate=24000):
    
    wav = torch.tensor(load_wav(str(generated_output_fp))).to(device)
    wavs_batch = wav.unsqueeze(0) # convert to batch format
    
    if sample_rate != chord_model._sample_rate:
        resampled_audio = resample(
            wavs_batch,
            orig_freq=sample_rate,
            new_freq=chord_model._sample_rate,
        )
    else:
        resampled_audio = wavs_batch

    with torch.no_grad():
        chord_label = chord_model.predict_step(
            batch=(resampled_audio, None),
            batch_idx=0,
        )[0]
    
    probs = [x[-1] if x[2] != "N" else 0 for x in chord_label]
    weights = [x[1]-x[0] for x in chord_label]
    chord_prob = sum(d * w for d, w in zip(probs, weights)) / sum(weights)   # weighted by the duration of each chord
    
    return chord_prob
    

def run_chordprob_metrics(chord_model, generated_output_fps, device='cuda', sample_rate=24000):
    
    # results = []
    for idx, generated_output_fp in enumerate(generated_output_fps):
        chord_prob = run_one_sample(chord_model, generated_output_fp, device, sample_rate)
        # results.append([generated_output_fp.name, "%.4f" % chord_prob])
        
        metadata_fp = str(generated_output_fp).replace('generated.wav', 'metadata.json')
        update_json(metadata_fp, { 'chordprob': round(chord_prob, 3)})
    

# have this function for offline test, not sure if there is better way
def test_run(input_dir):
    
    filenames = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    if not len(filenames):
        filenames = glob.glob(os.path.join(input_dir, "**/*.mp3"), recursive=True)
    filenames.sort()
    
    chord_ckpt = "hdfs:///home/byte_speech_sv/bigmusic/models/reranker/semi-sup-chord_epoch=103-step=56300.ckpt"
    hpath = chord_ckpt
    cache_dir = ".module_cache/musiclm"
    local_rank = 0
    chord_model = init_chord(hpath, local_rank, cache_dir)["chord"]
    
    results = []
    for filename in filenames:
        print (filename)
        chord_prob = run_one_sample(chord_model, filename, device='cuda', sample_rate=24000)
        results.append([os.path.splitext(os.path.basename(filename))[0], chord_prob])
    probs = [x[1] for x in results]
    results.append(["average", np.mean(probs)])
    filename_result = os.path.join(input_dir,  "chord_prob_metric.txt")
    with open(str(filename_result), "w") as fid:
        for item in results:
            fid.write("%s: %.4f\n" % (item[0], item[1]))
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Add arguments for the two directory paths
    parser.add_argument("--input_dir", type=str, help="Path to the wav directory")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    test_run(input_dir)
    
            
    
        
        