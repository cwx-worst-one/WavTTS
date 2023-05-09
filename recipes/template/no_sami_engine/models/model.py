import torch.nn as nn


class Model(nn.Module):
    pass


def get_model(model_name):
    if model_name == "model":
        return Model()
    else:
        raise NotImplementedError
