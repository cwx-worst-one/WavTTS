''' help function for grad traverse. '''
import torch


def recursive_traverse_grad_fn(fn, seen_fns, seen_params):
    '''tranverse a grad fn recursively.'''
    if fn in seen_fns:
        return
    seen_fns.add(fn)

    # record tensors
    if hasattr(fn, 'variable') and isinstance(fn.variable, torch.nn.Parameter):
        seen_params.add(fn.variable)

    # recursively tranverse
    if hasattr(fn, 'next_functions'):
        for u in fn.next_functions:
            if u[0] is not None:
                recursive_traverse_grad_fn(u[0], seen_fns, seen_params)
    if hasattr(fn, 'saved_tensors'):
        for t in fn.saved_tensors:
            recursive_traverse_grad_fn(t, seen_fns, seen_params)


def find_parameters(tensors):
    '''find paramters in the autograd graph for this tensor.'''
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    grad_fns = set()
    params = set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        recursive_traverse_grad_fn(tensor.grad_fn, grad_fns, params)
    return params
