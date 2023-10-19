import os, sys
import torch
import numpy as np
from sklearn import datasets
from openTSNE import TSNE
 
import tsneutil
 
import matplotlib.pyplot as plt
import seaborn as sns
import random

def _repara(stats):
    m, logs = torch.split(stats, 32, dim=-1)
    z = m + torch.randn_like(m) * torch.exp(logs)
    return z

# iris = datasets.load_iris()
# x, y = iris["data"], iris["target"]

# print("x: ", x.shape)
# print("y: ", y.shape, y)
# exit()
zh_feature_dir = sys.argv[1]
en_feature_dir = sys.argv[2]
suffixes = sys.argv[3]
plot_dir = sys.argv[4]

os.makedirs(plot_dir, exist_ok=True)

zh_feature_names = os.listdir(zh_feature_dir)
zh_feature_bn_prenet_paths = [os.path.join(zh_feature_dir, x) for x in zh_feature_names if x.endswith('_{}.npy'.format(suffixes))]

en_feature_names = os.listdir(en_feature_dir)
en_feature_bn_prenet_paths = [os.path.join(en_feature_dir, x) for x in en_feature_names if x.endswith('_{}.npy'.format(suffixes))]

plot_bn_prenet = True

### bn_prenet
if plot_bn_prenet:
    zh_bn_prenets = None
    for zh_feature_bn_prenet_path in zh_feature_bn_prenet_paths:
        bn_prenet = np.load(zh_feature_bn_prenet_path)
        bn_prenet = torch.from_numpy(bn_prenet)
        if zh_bn_prenets is None:
            zh_bn_prenets = bn_prenet
        else:
            zh_bn_prenets = np.concatenate((zh_bn_prenets, bn_prenet), axis=0)
    zh_labels = np.ones(zh_bn_prenets.shape[0]) * 0
    print("zh_bns: ", zh_bn_prenets.shape)

    en_bn_prenets = None
    for en_feature_bn_prenet_path in en_feature_bn_prenet_paths:
        bn_prenet = np.load(en_feature_bn_prenet_path)
        bn_prenet = torch.from_numpy(bn_prenet)
        if en_bn_prenets is None:
            en_bn_prenets = bn_prenet
        else:
            en_bn_prenets = np.concatenate((en_bn_prenets, bn_prenet), axis=0)
    en_labels = np.ones(en_bn_prenets.shape[0]) * 1
    print("en_bns: ", en_bn_prenets.shape)

    x = np.concatenate((zh_bn_prenets, en_bn_prenets), axis=0)
    y = np.concatenate((zh_labels, en_labels), axis=0)
    tsne = TSNE(
        perplexity=50,
        n_iter=500,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
    )
    embedding = tsne.fit(x)
    tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS, out_path='{}/{}.png'.format(plot_dir, suffixes))

    x = zh_bn_prenets
    y = [random.choice([0, 1]) for _ in range(zh_labels.shape[0])]
    y = np.asarray(y)
    tsne = TSNE(
        perplexity=50,
        n_iter=500,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
    )
    embedding = tsne.fit(x)
    tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS, out_path='{}/{}_zh_mix.png'.format(plot_dir, suffixes))

    x = en_bn_prenets
    y = [random.choice([0, 1]) for _ in range(en_labels.shape[0])]
    y = np.asarray(y)
    tsne = TSNE(
        perplexity=50,
        n_iter=500,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
    )
    embedding = tsne.fit(x)
    tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS, out_path='{}/{}_en_mix.png'.format(plot_dir, suffixes))
