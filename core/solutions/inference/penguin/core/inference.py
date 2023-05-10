#!/usr/bin/python3
from abc import ABCMeta


class InferSession:
    def __init__(self, config):
        self.inputs_shape_ = []
        self.outputs_shape_ = []
        self.inputs_ = []
        self.outputs_ = []

    def feed(self, inputs):
        self.inputs_.append(inputs)

    def fetch(self, output_index):
        if output_index >= len(self.outputs_):
            return None
        return self.outputs_[output_index]


class Inference(metaclass=ABCMeta):
    def __init__(self, config):
        raise NotImplementedError('Abstract method of %s' % self)

    def load_model(self):
        raise NotImplementedError('Abstract method of %s' % self)

    def run(self):
        raise NotImplementedError('Abstract method of %s' % self)
