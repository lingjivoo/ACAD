import torch
from torch.autograd.function import InplaceFunction
import numpy as np


class Dropout(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, prob, train=False, inplace=False):
        ctx.train = train
        ctx.inplace = inplace

        if not ctx.train:  # ctx.p == 0 or
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        ctx.noise_backward = cls._make_noise(input)
        for dim in range(input.size(0)):
            mask = np.random.binomial(1, 1-prob[dim].detach().cpu().numpy(), input.size(1))
            ctx.noise_backward[dim] = torch.tensor(mask)
        output.mul_(ctx.noise_backward)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:  # ctx.p > 0 and
            return grad_output * ctx.noise_backward, None, None, None, None  # * ctx.noise_backward
        else:
            return grad_output, None, None, None, None

