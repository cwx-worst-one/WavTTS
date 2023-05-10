'''
This module has two main goals.
  1. assure alignment between the algorithm code and online service.
  2. reduce develop cost and improve efficiency of most guy.

So, the below will be implemented:
  1. easy-use and automated solution to find any disalignment.
  2. easily unterstood inference implementation and documentation.
  3. etc.
'''

from .infer import (
    BaseInfer,
    INFERS,
    Factory,
    convert_to_tensor,
    convert_to_np,
    concat_global_states,
)
