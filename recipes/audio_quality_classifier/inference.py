# from sklearn.metrics import r2_score

# x = [
#     3.34,
#     3,
#     3,
#     2,
#     2,
#     1.83,
#     1.67,
#     1.67,
#     1.17,
#     1.33,
# ]
# y1 = [
#     -0.8639,
#     -11.41,
#     2.317,
#     2.94,
#     12.15,
#     7.89,
#     1.73,
#     0.867,
#     -0.6995,
#     8.44,
# ]
# y2 = [
#     0.31,
#     -3.72,
#     0.66,
#     0.28,
#     0.76,
#     1.50,
#     0.78,
#     1.90,
#     2.14,
#     0.92,
# ]
# y3 = [
#     -0.6364,
#     -4.5344,
#     -9.9382,
#     -7.7585,
#     0.3525,
#     -10.9343,
#     -13.1771,
#     -17.5407,
#     -8.2410,
#     -12.2118,
# ]

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# # Reshape y1 since it's a single feature
# y1_reshaped = [[i] for i in y3]

# # Fit the linear regression model
# model = LinearRegression().fit(y1_reshaped, x)

# # Predict y values using y1
# y_pred = model.predict(y1_reshaped)

# # Compute the R^2 score
# r2 = r2_score(x, y_pred)

# print(f"R^2 Score: {r2}")


# assert 1==2

import os
import torch
import soundfile as sf
from recipes.audio_quality_classifier.models.audio_quality_model.utils import init_audio_quality_classifier, aq_classifier_inference
from collections import OrderedDict
import glob

model = init_audio_quality_classifier(
    '/opt/tiger/arnold_experiment/samantha/logs/aq/stage_2/checkpoints/aq-step=002400-val_loss=0.381703.ckpt',
    0,
    None
)


data_path = '/opt/tiger/arnold_experiment/aq_test'
audio_paths = sorted(glob.glob(f'{data_path}/*.wav'))

for audio_path in audio_paths:
    audio, sr = sf.read(audio_path)
    audio = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
    print(audio.shape)
    output = aq_classifier_inference(model, audio, None)
    print(os.path.basename(audio_path), output)