import numpy as np
import os
from tqdm import tqdm
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from sklearn.cluster import MiniBatchKMeans, KMeans

import multiprocessing
import time

trainset = np.load(
    # "/mnt/bn/dguo-mir-workspace/Experiments/mulan_embeds/mulan1b_g4_170/embeds_mulan1b_g4_170.npy"
    # "/mnt/bd/dguo-mars/Dataset/aigc/mulan_embeds_gpt_all/embeds_mulan1b_gpt_all_127.npy"
    "/mnt/bd/dguo-mars/Dataset/aigc/mulan_embeds_md/gpt_text_embeds_mulan1b_md_114.npy"
)
np.random.seed(2023)
np.random.shuffle(trainset)
print(f"There are totally {len(trainset)} samples")
testset, trainset = np.split(trainset, [4096])
# trainset = trainset[::2, :]

TEST_LENGTH = testset.shape[0]
TRAIN_LENGTH = trainset.shape[0]
print(f"Train size = {TRAIN_LENGTH}")
print(f"Test size = {TEST_LENGTH}")

LOCAL_FOLDER = (
    "/mnt/bd/dguo-mars/Dataset/aigc/mulan_embeds_md/rvq_codebook/"
)
os.makedirs(LOCAL_FOLDER, exist_ok=True)

def find_codebook(
    trainset,
    testset,
    codebook_size=1024,
    codebook_nums=12,
    MAX_EXTENSION=1,
):


    # TEST_LENGTH = 1000
    # TRAIN_LENGTH = len(mulan_paths) - TEST_LENGTH
    batch_size = 512 * 1024
    # num_workers = 20

    ######### test
    # TRAIN_LENGTH = 1100-TEST_LENGTH
    # batch_size = 10
    # codebook_size=32
    ############
    print("TRAINSET: ", TRAIN_LENGTH)
    print("TESTSET: ", TEST_LENGTH)
    print("batch size:", batch_size)
    print("codebook size", codebook_size)

    trainset = trainset * MAX_EXTENSION
    print(trainset[:16, :4])

    testset = testset * MAX_EXTENSION
    print(testset[:16, :4])

    print(
        "codebook idx:",
        -1,
        " train error:",
        np.sum(trainset**2) / trainset.shape[0],
        " test  error:",
        np.sum(testset**2) / testset.shape[0],
    )
    with open(
        LOCAL_FOLDER + "kmeans_rvq_20M_bs50W_init2.log",
        "w",
    ) as fp:
        for codebook_idx in range(codebook_nums):
            start_time = time.time()
            # kmeans
            print("train kmeans: ", codebook_idx)
            kmeans = MiniBatchKMeans(
                init="k-means++",
                n_clusters=codebook_size,
                batch_size=batch_size,
                n_init=2,
                max_no_improvement=10,
                reassignment_ratio=0.01,
                verbose=1,
            ).fit(trainset)

            print("Inferring the residual and calculating loss")

            cluster_ids_x = kmeans.labels_
            codebook_i = kmeans.cluster_centers_
            for rx in range(0, len(trainset), 2048):
                ry = min(rx + 2048, len(trainset))
                trainset[rx:ry, :] = (
                    trainset[rx:ry, :] - codebook_i[cluster_ids_x[rx:ry], :]
                )
            error = np.sum(trainset**2)
            error = error / trainset.shape[0]

            testset_id = kmeans.predict(testset)
            testset = testset - codebook_i[testset_id, :]
            error_test = np.sum(testset**2)
            error_test = error_test / testset.shape[0]

            msg = [
                "codebook idx: " + repr(codebook_idx),
                " train error: " + repr(error),
                " test error: " + repr(error_test),
            ]
            fp.write(", ".join(msg) + "\n")
            print(
                "codebook idx:",
                codebook_idx,
                " train error:",
                error,
                " test error:",
                error_test,
            )
            print("used time:", time.time() - start_time)
            codebook_i = codebook_i / MAX_EXTENSION

            file_name = (
                "kmeans_minibatch_codebook_"
                + str(codebook_idx)
                + "_20M_bs50W_ninit2.npy"
            )
            file_name = LOCAL_FOLDER + file_name
            np.save(
                file_name,
                codebook_i,
            )

if __name__ == "__main__":
    find_codebook(trainset, testset)
