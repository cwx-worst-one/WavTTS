class Processor:
    def __init__(self):
        self.inference = None

    def process(self, message):
        result = self.get_result(message)
        return result

    def get_result(self, message):
        raise NotImplementedError(' Abstract method of %s' % self)

    def get_input(self):
        return self.inference.get_input()

    def get_input_all(self):
        return self.inference.get_input_all()

    def get_output(self):
        return self.inference.get_output()

    def get_output_all(self):
        return self.inference.get_output_all()
