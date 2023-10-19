import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function.
    In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    """
    Gradient Reversal Layer
    Code from:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class AdversarialPredictor(nn.Module):
    """
    AdversarialClassifier
        - 1 gradident reversal layer
        - n hidden linear layers with ReLU activation
        - 1 output linear layer with Softmax activation
    """
    def __init__(self, in_dim, out_dim, hidden_dims=[256], rev_scale=1):
        """
        Args:
             in_dim: input dimension
            out_dim: number of units of output layer (number of classes)
        hidden_dims: number of units of hidden layers
          rev_scale: gradient reversal scale
        """
        super(AdversarialPredictor, self).__init__()

        self.gradient_rev = GradientReversal(rev_scale)

        in_sizes = [in_dim] + hidden_dims[:]
        out_sizes = hidden_dims[:] + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, out_sizes)])

        self.activations = [nn.ReLU()] * len(hidden_dims) + [nn.Softmax(dim=-1)]

    def forward(self, x):
        x = self.gradient_rev(x)
        for (linear, activate) in zip(self.layers, self.activations):
            x = activate(linear(x))
        return x