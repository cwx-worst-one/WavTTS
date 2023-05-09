from samantha.stage import BaseStage


class Stage1(BaseStage):
    def __init__(self, takes, provides, serialize_opts=None):
        super().__init__(takes, provides, serialize_opts)
        raise NotImplementedError

    def _compute(self, x):
        raise NotImplementedError


class Stage2(BaseStage):
    def __init__(self, takes, provides, serialize_opts=None):
        super().__init__(takes, provides, serialize_opts)
        raise NotImplementedError

    def _compute(self, x):
        raise NotImplementedError
