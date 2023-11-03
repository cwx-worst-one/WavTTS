from contextlib import contextmanager


@contextmanager
def evaluate_model(model):
    _training = model.training
    try:
        model.eval()
        yield model
    finally:
        if _training:
            model.train()
