from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from ..components.attention import MultiHeadAttention, SeerAttention


class BaseModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.hooks = []

    def init_cache(
        self, init_values: Optional[torch.Tensor] = None, cache: Optional[dict] = None
    ) -> Tuple[dict, List]:
        """The `MultiHeadAttention` module optionally accepts `kv_cache` which
        stores the key and value tensors calculated for the previous positions.
        This method returns a dictionary that stores all caches, and the necessary
        hooks for the key and value projection modules that save the intermediate
        tensors to be reused during later calculations.

        The `self.hooks` contain a list of PyTorch RemovableHandle objects to stop
        the hooks from being called. This is done in `self.deinit_cache`.

        Args:
            init_values (Optional[torch.Tensor], optional):
                Tensor containing values to initialize the k/v cache with
                (i.e., single forward pass). Defaults to None.

            cache (Optional[dict], optional):
                Existing k/v cache. Defaults to None.

        Returns:
            Tuple[dict, List]:
                A dictionary object mapping the key/value projection modules
                to its cache
        """

        cache = {**cache} if cache is not None else {}

        def save_to_cache(module, _, output):
            if module not in cache:
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, (MultiHeadAttention, SeerAttention)):
                layer._use_cache = True
                self.hooks.append(layer.to_k.register_forward_hook(save_to_cache))
                self.hooks.append(layer.to_v.register_forward_hook(save_to_cache))

        self.apply(install_hooks)

        if init_values is not None:
            self.forward(init_values)

        return cache

    def deinit_cache(self) -> None:
        def unset_cache(layer: nn.Module):
            if isinstance(layer, (MultiHeadAttention, SeerAttention)):
                layer._use_cache = False

        self.apply(unset_cache)

        for h in self.hooks:
            h.remove()
        self.hooks = []
