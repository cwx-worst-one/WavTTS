import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pylab as plt


class HParams():
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
    with open(file, 'r') as f:
        hp = yaml.safe_load(f)
    hp = HParams(**hp)
    return hp


def remove_ddp_module(ckpt):
    from collections import OrderedDict
    new_dict = OrderedDict()
    for key in ckpt:
        new_key = key.replace('module.', '', 1)
        new_dict[new_key] = ckpt[key]
    return new_dict


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram,
                   aspect="auto",
                   origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig
