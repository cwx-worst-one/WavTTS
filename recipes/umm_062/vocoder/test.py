import yaml
import torch
from easydict import EasyDict

from recipes.umm_062.vocoder.bigvgan import BigVGAN
from recipes.umm_062.vocoder.audio_encoder import AudioEncoder


def backward_receptive_field(model, dim=1, length=1000*300, device='cpu'):
    model = model.to(device).eval()

    x = torch.randn(size=[1, dim, length]).to(device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)
    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    print(torch.nonzero(gradmap).squeeze().detach().cpu().numpy())
    rf = (gradmap != 0).sum()
    print("Backward receptive field: {}".format(rf))
    return

hp = EasyDict(yaml.load(open('recipes/umm/vocoder/hyper.yaml').read()))
model = BigVGAN(hp)
backward_receptive_field(model, dim=64, length=300, device='cuda')

model = AudioEncoder()
backward_receptive_field(model, dim=1, length=1000*300, device='cuda')


from flow import ResidualCouplingBlock
model = ResidualCouplingBlock(
    channels=64,
    hidden_channels=256,
    kernel_size=5,
    dilation_rate=1,
    n_layers=4,
    n_flows=4,
    affine=False,
).eval().cuda()
x = torch.randn(size=[1, 64, 100]).cuda()
y = model(x)[0]
z = model(y, reverse=True)[0]

print((x - y).abs().max())
print((x - z).abs().max())
