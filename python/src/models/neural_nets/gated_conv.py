import torch
import torch.nn.functional as F
from torch import nn


class GatedConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.mask_conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.conv3d(input)
        mask = self.sigmoid(self.mask_conv3d(input))

        return output, mask


class GatedConv3dWithActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: nn.Module,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        self.activation = activation

        self.conv = GatedConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, mask = self.conv(input)
        if self.activation is not None:
            output = self.activation(output)
        return mask * output
