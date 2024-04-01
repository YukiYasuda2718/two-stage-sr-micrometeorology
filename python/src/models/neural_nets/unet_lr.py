from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.neural_nets.gated_conv import GatedConv3dWithActivation
from src.models.neural_nets.voxel_shuffle import VoxelShuffle

logger = getLogger()


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        stride: int,
        n_layers_in_block: int,
    ):
        super().__init__()

        assert n_layers_in_block >= 1

        layers = [
            GatedConv3dWithActivation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
                activation=nn.ReLU(),
            )
        ]

        for _ in range(n_layers_in_block - 1):
            layers.append(
                GatedConv3dWithActivation(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                    activation=nn.ReLU(),
                )
            )

        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


def ICNR(tensor: torch.Tensor, scale_factor: int, initializer=nn.init.kaiming_uniform_):
    OUT, IN, D, H, W = tensor.shape
    sub = torch.zeros(OUT // (scale_factor**3), IN, D, H, W)
    sub = initializer(sub)

    kernel = torch.zeros_like(tensor)
    for i in range(OUT):
        kernel[i] = sub[i // (scale_factor**3)]

    return kernel


class UpBlock(nn.Module):
    def __init__(
        self,
        in1_channels: int,
        in2_channels: int,
        out_channels: int,
        bias: bool,
        n_layers_in_block: int,
        upfactor: int,
    ):
        super().__init__()

        assert n_layers_in_block >= 1

        layers = [
            nn.Conv3d(
                in_channels=in1_channels + in2_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.LeakyReLU(),
        ]

        for _ in range(n_layers_in_block - 1):
            layers.append(
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=bias,
                )
            )
            layers.append(nn.LeakyReLU())

        self.convs = nn.Sequential(*layers)

        self.up = nn.Sequential(
            nn.Conv3d(
                in_channels=in1_channels,
                out_channels=in1_channels * (upfactor**3),
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.LeakyReLU(),
            VoxelShuffle(factor=upfactor),
        )
        kernel = ICNR(self.up[0].weight, scale_factor=upfactor)
        self.up[0].weight.data.copy_(kernel)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x3 = self.up(x1)

        y = torch.cat([x2, x3], dim=1)  # concat along channel dim

        return self.convs(y)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, bias: bool, dropout: float):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
        )
        self.act = nn.LeakyReLU()
        self.last_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_dropout(self.act(x + self.convs(x)))


class UNetLr(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_feat0: int,
        num_feat1: int,
        num_feat2: int,
        num_feat3: int,
        num_latent_blocks: int,
        bias: bool,
        n_layers_in_block: int,
        interpolation_mode: str,
        lr_size: list[int],
        dropout: float,
        **kwargs,
    ):
        super().__init__()
        logger.info("UNetLr (U-Net 1)")
        logger.info(f"dropout = {dropout}")

        self.lr_size = tuple(lr_size)  # z, y, x

        # z, y, x sizes, where y and x sizes are the same.
        # Both input and output have lr_size.
        assert self.lr_size == (10, 80, 80), f"Size {lr_size} is not supported."

        self.resized_shape = (8, 80, 80)
        self.mode = interpolation_mode

        # `+ 1` in channel is necessary to concatenate with building data
        self.conv0 = GatedConv3dWithActivation(
            in_channels=in_channels + 1,
            out_channels=num_feat0,
            kernel_size=3,
            padding=1,
            bias=bias,
            activation=None,
        )

        self.down_stride2 = nn.AvgPool3d(kernel_size=2, stride=2)

        # `+ 1` in channel is necessary to concatenate with building data
        self.down1 = DownBlock(
            in_channels=num_feat0 + 1,
            out_channels=num_feat1,
            bias=bias,
            stride=2,
            n_layers_in_block=n_layers_in_block,
        )
        self.down2 = DownBlock(
            in_channels=num_feat1 + 1,
            out_channels=num_feat2,
            bias=bias,
            stride=2,
            n_layers_in_block=n_layers_in_block,
        )
        self.down3 = DownBlock(
            in_channels=num_feat2 + 1,
            out_channels=num_feat3,
            bias=bias,
            stride=2,
            n_layers_in_block=n_layers_in_block,
        )

        # `+ 1` in channel is necessary to concatenate with building data
        latent_layers = [
            nn.Conv2d(
                in_channels=num_feat3 + 1,
                out_channels=num_feat3,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
        ]
        for _ in range(num_latent_blocks):
            latent_layers.append(
                ResBlock(in_channels=num_feat3, bias=bias, dropout=dropout)
            )

        self.latent_layers = nn.Sequential(*latent_layers)

        # `+ 1` in channel is necessary to concatenate with building data
        self.up3 = UpBlock(
            in1_channels=num_feat3 + 1,
            in2_channels=num_feat2 + 1,
            out_channels=num_feat2,
            bias=bias,
            upfactor=2,
            n_layers_in_block=n_layers_in_block,
        )
        self.up2 = UpBlock(
            in1_channels=num_feat2 + 1,
            in2_channels=num_feat1 + 1,
            out_channels=num_feat1,
            bias=bias,
            upfactor=2,
            n_layers_in_block=n_layers_in_block,
        )
        self.up1 = UpBlock(
            in1_channels=num_feat1 + 1,
            in2_channels=num_feat0 + 1,
            out_channels=num_feat0,
            bias=bias,
            upfactor=2,
            n_layers_in_block=n_layers_in_block,
        )

        self.last = nn.Sequential(
            nn.Conv3d(
                in_channels=num_feat0 + in_channels + 1,
                out_channels=num_feat0,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            nn.LeakyReLU(),
            nn.Conv3d(
                in_channels=num_feat0,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def forward(
        self, x: torch.Tensor, b: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        #
        assert x.shape[-3:] == b.shape[-3:] == self.lr_size

        # from (10, 80, 80) to (8, 80, 80)
        x0 = F.interpolate(x, size=self.resized_shape, mode=self.mode)
        b0 = F.interpolate(b, size=self.resized_shape, mode=self.mode)

        xb = torch.cat([x0, b0], dim=1)
        y0 = self.conv0(xb)
        y0 = torch.cat([y0, b0], dim=1)

        y1 = self.down1(y0)
        b1 = self.down_stride2(b0)
        y1 = torch.cat([y1, b1], dim=1)

        y2 = self.down2(y1)
        b2 = self.down_stride2(b1)
        y2 = torch.cat([y2, b2], dim=1)

        y3 = self.down3(y2)
        b3 = self.down_stride2(b2)
        y = torch.cat([y3, b3], dim=1)

        # D = 1, so depth dim can be dropped
        # Then, y are 2D images
        y = y.squeeze(2)
        y = self.latent_layers(y)

        # Re-convert to 3D tensors
        y = y.unsqueeze(2)
        y = y + y3
        y = torch.cat([y, b3], dim=1)

        y = self.up3(y, y2)
        y = torch.cat([y, b2], dim=1)

        y = self.up2(y, y1)
        y = torch.cat([y, b1], dim=1)

        y = self.up1(y, y0)
        y = F.interpolate(y, size=self.lr_size, mode=self.mode)

        y = torch.cat([y, x, b], dim=1)  # concat along channel dim

        return self.last(y)
