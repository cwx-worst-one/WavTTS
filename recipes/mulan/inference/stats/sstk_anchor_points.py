import numpy as np

def load_anchor_points_from_mulan_ckpt(mulan_hpath):
    if 'mulan-step=005000-median_rank_1=72-kaggle' in mulan_hpath:
        npy_path = 'recipes/mulan/inference/stats/binary_center_mulan_72.npy'
    elif 'mulan-step=005500-median_rank_1=63-kaggle' in mulan_hpath:
        npy_path = 'recipes/mulan/inference/stats/binary_center_mulan_63.npy'
    else:
        print(
            'WARNING: load_anchor_points_from_mulan_ckpt could not find associated anchor points.',
            'Using default anchor for mulan path:', mulan_hpath
        )
        npy_path = 'recipes/mulan/inference/stats/binary_center_mulan_72.npy'
    return load_anchor_points(npy_path)

def load_anchor_points(npy_path='recipes/mulan/inference/stats/binary_center_mulan_72.npy'):
    return np.load(npy_path)