from typing import Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class CausalConv2D(nn.Conv2d):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be set as None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self,
        x,
    ):
        x = F.pad(
            x,
            pad=(
                self._left_padding,
                self._right_padding,
                self._left_padding,
                self._right_padding,
            ),
        )
        x = super().forward(x)
        return x


class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif (
                isinstance(padding, list)
                and len(padding) == 2
                and padding[0] + padding[1] == kernel_size - 1
            ):
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache


class MaskedConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mask: torch.Tensor,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
    ):
        """
        2d masked convolution
        mask [torch.Tensor]: Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                the convolution should be masked, and 1s otherwise.
        """
        super().__init__()
        kernel_size = (mask.shape[0], mask.shape[1])

        if padding is None:
            # pad same
            padding = tuple(
                [dilation * (kernel_size[i] - stride) // 2 for i in range(2)]
            )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.register_buffer("mask", mask[None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # zero at masked positions
        self.conv.weight.data *= self.mask
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        mask_center=False,
    ):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(in_channels, out_channels, mask, stride, dilation)


class HorizontalStackConvolution(MaskedConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        mask_center=False,
    ):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(in_channels, out_channels, mask, stride, dilation)


class GatedMaskedConv(nn.Module):
    def __init__(
        self, in_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1
    ):
        super().__init__()
        self.conv_vert = VerticalStackConvolution(
            in_channels,
            2 * in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.conv_horiz = HorizontalStackConvolution(
            in_channels,
            2 * in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.conv_vert_to_horiz = nn.Conv2d(
            2 * in_channels, 2 * in_channels, kernel_size=1, padding=0
        )

        self.conv_horiz_1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, padding=0
        )

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)

        # gate horizontal/vertical feature
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)

        # residual, skip due to subsampling
        # h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class CausalGatedConv2d(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, kernel_size: int
    ):

        super().__init__()

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(
            1, hidden_dim, kernel_size, mask_center=True
        )
        self.conv_hstack = HorizontalStackConvolution(
            1, hidden_dim, kernel_size, mask_center=True
        )

        self.conv_layers = nn.ModuleList(
            [
                GatedMaskedConv(hidden_dim, kernel_size, stride=2),
                # TODO: ReLU?
                GatedMaskedConv(hidden_dim, kernel_size, stride=2),
                # TODO: ReLU?
            ]
        )
        self.proj = nn.Sequential(
            Rearrange("b c t f -> b t (c f)"), nn.Linear(input_dim * 64, output_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (b, c, t, f)

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)

        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        return self.proj(h_stack)

    def get_flops(self, b, t, d):
        return 1.0  # TODO xD
