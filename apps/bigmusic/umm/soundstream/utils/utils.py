import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_config_from_file(file):
    with open(file, "r") as f:
        hp = yaml.safe_load(f)
    hp = HParams(**hp)
    return hp


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
    return


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def export_tensor(data, name):
    print("export {} shape {}".format(name, data.shape))
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    np.save(name, data)
    WriteMatrixToBinary("{}.bin".format(name), data)


def set_random_seed(seed=123):
    """Set random seed manully to get deterministic results"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
