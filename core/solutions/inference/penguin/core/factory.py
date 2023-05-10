from core.solutions.inference.penguin.core.register import Registers
from core.solutions.inference.infer import Factory as DolphinFactory


class Factory:
    inference_dist = {}
    processor_dist = {}

    def __init__(self):
        pass

    @staticmethod
    def get_inference(model_name, config):
        inference = DolphinFactory.get_inference(model_name, config)
        inference.numpy_out = True
        return inference

    @staticmethod
    def get_processor(model_name, config):
        if Factory.processor_dist.get((model_name, config), None) == None:
            Factory.processor_dist[(model_name, config)] = Registers.processor[model_name](config)
        return Factory.processor_dist[(model_name, config)]

    @staticmethod
    def get_new_processor(model_name, config):
        Factory.processor_dist[(model_name, config)] = Registers.processor[model_name](config)
        return Factory.processor_dist[(model_name, config)]
