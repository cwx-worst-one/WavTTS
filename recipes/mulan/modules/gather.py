import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""
    @staticmethod
    def forward(ctx, input):
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        output = torch.stack(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        grad_output = torch.cat(grad_output)
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, async_op=False)
        return grad_output[dist.get_rank()], None
