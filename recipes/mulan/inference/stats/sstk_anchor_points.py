import numpy as np

def load_anchor_points(npy_path='recipes/mulan/inference/stats/binary_center.npy'):
    return np.load(npy_path)