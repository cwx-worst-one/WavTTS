import logging


class ModelTester:
    def __init__(self, args):
        self.args = args
        self.model_dict = {}
        self.action = args.benchmark_params.action

    def gen_train_config(self, provider):
        raise NotImplementedError

    def gen_generate_config(self, provider):
        raise NotImplementedError

    def load_from_pretrained(self, provider, checkpoint_path):
        raise NotImplementedError

    def gen_config(self, provider):
        if self.action == 'train':
            return self.gen_train_config(provider)
        elif self.action == 'generate':
            return self.gen_generate_config(provider)
        else:
            logging.warning(
                f'{__name__}: action {self.action} has not been implemented.')
            exit(0)

    def gen_model(self, provider):
        raise NotImplementedError

    def gen_inputs(self, provider):
        raise NotImplementedError

    def get_model_size(self, provider, model):
        model_size = 0
        if model == None:
            return model_size

        model_size = sum(p.numel() for p in model.parameters())

        return model_size
