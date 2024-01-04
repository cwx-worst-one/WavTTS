import os
import random
import glob
import torch
import tqdm
import numpy as np
import soundfile as sf
from recipes.audio_quality_classifier.models.audio_quality_model.utils import init_audio_quality_classifier, aq_classifier_inference

data_slice = [0.8, 1.0]
asset_path = '/opt/tiger/arnold_experiment/samantha/aq_mos'
audio_paths = glob.glob(f'{asset_path}/*.wav')
random.seed(0)
random.shuffle(audio_paths)
start = int(len(audio_paths)*data_slice[0])
end = int(len(audio_paths)*data_slice[1])

model = init_audio_quality_classifier(
    '/opt/tiger/arnold_experiment/samantha/logs/aq/stage_2/checkpoints/aq-step=010320-val_loss=0.337832.ckpt',
    0,
    None
)

positive_scores = []
negative_scores = []
for audio_path in tqdm.tqdm(audio_paths[start:end]):
    _, gt_score = os.path.basename(audio_path)[:-4].split('_')
    try:
        audio = torch.from_numpy(sf.read(audio_path)[0]).float().cuda()[None, None, ]
        score = aq_classifier_inference(model, audio, None).cpu().numpy()
    except:
        continue
    if float(gt_score) > 2:
        print('positive', os.path.basename(audio_path), gt_score, score)
        positive_scores.append(score)
    else:
        print('negative', os.path.basename(audio_path), gt_score, score)
        negative_scores.append(score)

from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_f1(positive_scores, negative_scores, threshold):
    # Convert scores to binary labels based on threshold
    y_true = [1]*len(positive_scores) + [0]*len(negative_scores)
    y_pred = [1 if s >= threshold else 0 for s in positive_scores] + [1 if s >= threshold else 0 for s in negative_scores]

    return f1_score(y_true, y_pred)

# print(calculate_f1(positive_scores, negative_scores, -4.0))
# assert 1==2

best_threshold = 0.0
best_f1 = 0.0
    
# Test thresholds between 0 and 1 in increments of 0.01
for threshold in [i/2. for i in range(-30, 10,)]:
    f1 = calculate_f1(positive_scores, negative_scores, threshold)
    print(f"Threshold: {threshold}, F1: {f1}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best F1: {best_f1}")
print(f"Best Threshold: {best_threshold}")
    