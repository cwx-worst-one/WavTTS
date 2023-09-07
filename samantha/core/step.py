import enum


class Step(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"
    EXPORT = "export"
