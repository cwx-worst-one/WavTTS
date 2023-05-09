import a_unet
import a_unet.apex
import torch.nn as nn


class ArchinetUNet(nn.Module):
    def __init__(self, num_signal_channels=2):
        super().__init__()
        UNet = a_unet.TimeConditioningPlugin(a_unet.apex.XUNet)
        channels = [256, 512, 512]
        factors = [2, 2, 4]
        items = [2, 2, 2]
        sequences = (channels, factors, items)

        self.model = UNet(
            dim=1,
            in_channels=num_signal_channels,
            blocks=[
                a_unet.apex.XBlock(
                    channels=channels,
                    factor=factor,
                    items=[a_unet.apex.ModulationItem, a_unet.apex.ConvNextV2Item]
                    * n_items,
                )
                for channels, factor, n_items in zip(*sequences)
            ],
            modulation_features=512,
            resnet_groups=2,
        )

    def forward(self, audio, sigma):
        return self.model(audio, time=sigma.squeeze(1))
