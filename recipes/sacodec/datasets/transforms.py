
import torch

class AddConditionsTransform():
    def __init__(self, conditions=""):
        self.conditions = conditions

    def __call__(self, item):
        return { **item, 'conditions': self.conditions }

class AddDurationTransform():
    def __init__(self, duration):
        assert isinstance(duration, int)
        self.duration = duration

    def __call__(self, item):
        if 'duration' in item:
            durations = set()
            for d in item['duration']:
                durations.add(d.cpu().item() if isinstance(d, torch.Tensor) else d)
            assert len(durations) == 1, f"Duration in batch must be the same, got: {durations}"
            item['duration'] = list(durations)[0]
            return item
        else:
            return { **item, 'duration': self.duration }
