import os
from collections import defaultdict
from enum import Enum
from typing import Dict

import torch
from torch import nn


class ModuleTracker:
    def __init__(self, model: nn.Module, name="module", hints=None):
        self.model: nn.Module = model
        self.name = name

        self.registered = False
        self.steps = 0
        self.groundtruth = defaultdict(lambda: defaultdict(list))
        self.forward_hooks = {}
        self.children_names = {}

    def start(self):
        self._register_hooks()

    def stop(self):
        self._unregister_hooks()

    def export(self):
        return dict(map(lambda x: (x[0], dict(x[1])), self.groundtruth.items()))

    def _register_hooks(self, reset=True):
        def track_groundtruth(m, x, kw, y):
            if hasattr(y, "_fields"):
                y = y._asdict()
            n = self.children_names[m]
            self.groundtruth[self.steps][n].append(
                {"args": x, "kwargs": kw, "output": y}
            )
            if m is self.model:
                self.steps += 1

        def add_hooks(m: nn.Module):
            if self.registered:
                return
            self.forward_hooks[m] = m.register_forward_hook(
                track_groundtruth, with_kwargs=True
            )

        self.groundtruth = defaultdict(lambda: defaultdict(list))
        self.children_names = {
            m: ".".join([self.name, n]).strip(".")
            for n, m in self.model.named_modules()
        }
        self.model.apply(add_hooks)
        self.registered = True

    def _unregister_hooks(self):
        if not self.registered:
            return
        # reset model to original status
        for _, handler in self.forward_hooks.items():
            handler.remove()

        self.groundtruth = defaultdict(lambda: defaultdict(list))
        self.children_names.clear()
        self.forward_hooks.clear()
        self.registered = False


class Registry(object):
    registry: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    @classmethod
    def get(cls, name: str = None):
        name = name or "default"
        return cls.registry[name]

    @classmethod
    def clear(cls, name: str = None, clear_all=True):
        if clear_all:
            cls.registry.clear()
        else:
            name = name or "default"
            cls.registry[name].clear()
            del cls.registry[name]


class Status(Enum):
    STOP = 0
    RUNNING = 1


_STATUS = Status.RUNNING if os.getenv("ENABLE_GROUNDTRUTH") == "1" else Status.STOP


def enable():
    return _STATUS == Status.RUNNING


def start():
    global _STATUS
    _STATUS = Status.RUNNING


def stop():
    global _STATUS
    _STATUS = Status.STOP


def emit(name: str, data: Dict = None, **kwargs):
    if _STATUS != Status.RUNNING:
        return
    data = dict(data or {})
    data.update(kwargs)

    g = Registry.get(name)
    for k, v in data.items():
        g[k].append(v)


def export(name: str = None, export_all=True):
    if export_all:
        return {k: dict(v) for k, v in Registry.registry.items()}

    return dict(Registry.get(name))


def clear(name: str = None, clear_all=True):
    Registry.clear(name, clear_all)


####################################################
# main
####################################################

if __name__ == "__main__":
    emit("semantic", input=10)
    emit("semantic", output=20)

    groundtruth = export()
    print(f"{groundtruth=}")

    start()
    emit("diffusion", input=30)
    emit("diffusion", output=40)
    groundtruth = export("diffusion")
    clear()
    print(f"{groundtruth=}")

    class Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(1, 10)

        @torch.no_grad()
        def forward(self, x):
            return self.linear(x) + 1

    m = Module()
    tracker = ModuleTracker(m, "mod")
    tracker.start()
    for i in range(3):
        x = torch.tensor([i], dtype=torch.float)
        y = m(x)

    d = tracker.export()
    tracker.stop()

    # d = dict(map(lambda x: (x[0], dict(x[1])), d.items()))

    # torch.save()
    import pprint

    pprint.pprint(d)
