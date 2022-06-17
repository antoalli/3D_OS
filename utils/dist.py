import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gathers tensors from all processes, supporting backward propagation."""

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(inp) for _ in range(dist.get_world_size())]
            dist.all_gather(output, inp)
        else:
            output = [inp]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (inp,) = ctx.saved_tensors
        if dist.is_available() and dist.is_initialized():
            grad_out = torch.zeros_like(inp)
            grad_out[:] = grads[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_dist():
        return dist.get_rank()
    return 0


def get_ws():
    if is_dist():
        return dist.get_world_size()
    return 1
