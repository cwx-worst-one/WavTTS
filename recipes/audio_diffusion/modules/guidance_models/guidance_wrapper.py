import torch
import torch.nn as nn


class GuidanceWrapper(nn.Module):
    def __init__(
        self, guidance_model, guidance_target, guidance_weight, loss=torch.nn.MSELoss()
    ):
        super().__init__()
        self.guidance_model = guidance_model
        self.guidance_weight = guidance_weight
        self.guidance_target = guidance_target
        self.loss = loss

    def forward(self, x, denoised_x, sigma):
        pred_metric = self.guidance_model(denoised_x, sigma)
        loss = self.loss(pred_metric, self.guidance_target.expand(pred_metric.shape))
        grad = torch.autograd.grad(loss * self.guidance_weight, x)[0]
        return grad
