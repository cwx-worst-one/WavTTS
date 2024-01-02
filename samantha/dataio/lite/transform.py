class ItemTransformBase:
    def __init__(self):
        pass

    def __call__(self, item):
        raise NotImplementedError


class CollatorBase:
    def __init__(self):
        pass

    def __call__(self, batch):
        raise NotImplementedError
