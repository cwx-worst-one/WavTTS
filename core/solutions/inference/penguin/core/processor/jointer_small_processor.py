from core.solutions.inference.penguin.core.factory import Factory
from core.solutions.inference.penguin.core.register import Registers
from core.solutions.inference.penguin.core.processor.processor import Processor


@Registers.processor.register('jointer_small')
class JointerSmall(Processor):
    def __init__(self, config):
        self.inference = Factory.get_inference('jointer_small', config)
        pass

    def get_result(self, message):
        input_name = self.get_input()
        result = self.inference.run(input_data=[message[input_name[0]], message[input_name[1]]])
        return result
