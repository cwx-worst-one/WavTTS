import os
import random
import glob
import torch
import tqdm
import pandas as pd
import numpy as np
import soundfile as sf
from recipes.audio_quality_classifier.models.audio_quality_model.utils import init_audio_quality_classifier, aq_classifier_inference

data_slice = [0.8, 1.0]
asset_path = 'aq_dataset'

h_audio_paths = glob.glob(f'{asset_path}/high_quality/*.wav')
l_audio_paths = glob.glob(f'{asset_path}/low_quality/*.npy')
df = pd.read_csv(f'{asset_path}/dirty_kaggle.csv')
music_ids = df['music_id'].to_list()
l_audio_paths = [i for i in l_audio_paths if os.path.basename(i)[:-4] in music_ids]

random.seed(0)
random.shuffle(h_audio_paths)
h_start = int(len(h_audio_paths)*data_slice[0])
h_end = int(len(h_audio_paths)*data_slice[1])

random.seed(0)
random.shuffle(l_audio_paths)
l_start = int(len(l_audio_paths)*data_slice[0])
l_end = int(len(l_audio_paths)*data_slice[1])

model = init_audio_quality_classifier(
    '/opt/tiger/arnold_experiment/samantha/logs/aq/stage_2/checkpoints/aq-step=010320-val_loss=0.337832.ckpt',
    0,
    None
)

h_scores = []
for h_audio_path in tqdm.tqdm(h_audio_paths[h_start:h_end]):
    h_audio = torch.from_numpy(sf.read(h_audio_path)[0].T).mean(0, True).float()[None,].cuda()
    h_score = aq_classifier_inference(model, h_audio, None).cpu().numpy()
    h_scores.append(h_score)
    
l_scores = []
for l_audio_path in tqdm.tqdm(l_audio_paths[l_start:l_end]):
    l_audio = torch.tensor((np.load(l_audio_path) / 32768.0).astype("float32"))[None,].cuda()
    l_score = aq_classifier_inference(model, l_audio, None).cpu().numpy()
    l_scores.append(l_score)

print(np.mean(h_scores), np.mean(l_scores), np.median(h_scores), np.median(l_scores))

from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_f1(positive_scores, negative_scores, threshold):
    # Convert scores to binary labels based on threshold
    y_true = [1]*len(positive_scores) + [0]*len(negative_scores)
    y_pred = [1 if s >= threshold else 0 for s in positive_scores] + [1 if s >= threshold else 0 for s in negative_scores]

    return f1_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)

positive_scores = h_scores
negative_scores = l_scores
best_threshold = 0.0
best_f1 = 0.0
    
# Test thresholds between 0 and 1 in increments of 0.01
for threshold in [i/2. for i in range(-30, 10)]:
    f1, p, r  = calculate_f1(positive_scores, negative_scores, threshold)
    print(f"Threshold: {threshold}, F1: {f1} P: {p} R: {r}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best F1: {best_f1}")
print(f"Best Threshold: {best_threshold}")