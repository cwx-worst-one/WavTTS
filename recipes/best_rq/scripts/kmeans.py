import multiprocessing
import os
import time
from glob import glob

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

np.random.seed(2023)
bucket_size = 900000
fnames = glob("/mnt/bn/zongyu-lq/features/best_rq/step=011000/*.npy")[:10]
trainset = np.zeros((len(fnames) * bucket_size, 1024))
for i, fname in tqdm(enumerate(fnames)):
    trainset[i * bucket_size : (i + 1) * bucket_size] = np.load(fname)
np.random.shuffle(trainset)
testset, trainset = np.split(trainset, [int(trainset.shape[0] * 0.1)])

TEST_LENGTH = testset.shape[0]
TRAIN_LENGTH = trainset.shape[0]


def find_codebook(trainset, testset, codebook_size=1024, multiplier=1):

    batch_size = 256 * multiprocessing.cpu_count()

    print("TRAINSET: ", TRAIN_LENGTH)
    print("TESTSET: ", TEST_LENGTH)
    print("batch size:", batch_size)
    print("codebook size", codebook_size)

    trainset = trainset * multiplier
    testset = testset * multiplier

    start_time = time.time()
    # kmeans
    kmeans = MiniBatchKMeans(
        init="k-means++",
        n_clusters=codebook_size,
        batch_size=batch_size,
        max_iter=100,
        n_init=3,
        max_no_improvement=10,
        reassignment_ratio=0.003,
        verbose=1,
    ).fit(trainset)

    codebook = kmeans.cluster_centers_

    print("used time:", time.time() - start_time)

    codebook = codebook / multiplier
    file_name = "best_rq_kmeans_minibatch_codebook.npy"
    np.save(os.path.join("/mnt/bn/zongyu-lq/ckpts/best_rq", file_name), codebook)


if __name__ == "__main__":
    find_codebook(trainset, testset)
