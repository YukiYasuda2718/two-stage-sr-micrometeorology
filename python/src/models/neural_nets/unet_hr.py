from logging import getLogger
from typing import Literal

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
        activation_layer: nn.Module,
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
                activation=activation_layer(),
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
                    activation=activation_layer(),
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
        activation_layer: nn.Module,
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
            activation_layer(),
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
            layers.append(activation_layer())

        self.convs = nn.Sequential(*layers)

        self.up = nn.Sequential(
            nn.Conv3d(
                in_channels=in1_channels,
                out_channels=in1_channels * (upfactor**3),
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            activation_layer(),
            VoxelShuffle(factor=upfactor),
        )
        kernel = ICNR(self.up[0].weight, scale_factor=upfactor)
        self.up[0].weight.data.copy_(kernel)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x3 = self.up(x1)

        y = torch.cat([x2, x3], dim=1)  # concat along channel dim

        return self.convs(y)


def act_layer_maker(activation_type: str) -> nn.Module:
    if activation_type == "ReLU":
        return nn.ReLU
    elif activation_type == "LeakyReLU":
        return nn.LeakyReLU
    elif activation_type == "SiLU":
        return nn.SiLU
    else:
        raise ValueError(f"Activation type {activation_type} is not supported.")


class ResBlock(nn.Module):
    def __init__(
        self, in_channels: int, bias: bool, dropout: float, activation_layer: nn.Module
    ):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
            activation_layer(),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
        )
        self.act = activation_layer()
        self.last_dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_dropout(self.act(x + self.convs(x)))


class UNetHr(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lst_n_feats: list[int],
        scale_factor: int,
        num_latent_layers: int,
        bias: bool,
        n_layers_in_block: int,
        interpolation_mode: str,
        dropout: float,
        activation_type_down: Literal["ReLU", "LeakyReLU", "SiLU"],
        activation_type_latent: Literal["ReLU", "LeakyReLU", "SiLU"],
        activation_type_up: Literal["ReLU", "LeakyReLU", "SiLU"],
        **kwargs,
    ):
        super().__init__()

        logger.info("UNetHr (U-Net 2)")

        self.mode = interpolation_mode

        if scale_factor == 1:
            self.up0 = nn.Identity()
        else:
            self.up0 = nn.Upsample(scale_factor=scale_factor, mode=self.mode)

        # `+ 1` in channel is necessary to concatenate with building data
        self.conv0 = GatedConv3dWithActivation(
            in_channels=in_channels + 1,
            out_channels=lst_n_feats[0],
            kernel_size=3,
            padding=1,
            bias=bias,
            activation=None,
        )

        self.down_stride2 = nn.AvgPool3d(kernel_size=2, stride=2)

        # `+ 1` in channel is necessary to concatenate with building data
        downs = []
        for i in range(0, len(lst_n_feats) - 1):
            downs.append(
                DownBlock(
                    in_channels=lst_n_feats[i] + 1,
                    out_channels=lst_n_feats[i + 1],
                    bias=bias,
                    stride=2,
                    n_layers_in_block=n_layers_in_block,
                    activation_layer=act_layer_maker(activation_type_down),
                )
            )
        self.downs = nn.ModuleList(downs)

        # `+ 1` in channel is necessary to concatenate with building data
        latent_layers = []
        for i in range(num_latent_layers):
            latent_layers.append(
                ResBlock(
                    in_channels=lst_n_feats[-1] + 1,
                    bias=bias,
                    dropout=dropout,
                    activation_layer=act_layer_maker(activation_type_latent),
                )
            )
        self.latent_layers = nn.Sequential(*latent_layers)

        # `+ 1` in channel is necessary to concatenate with building data
        ups = []
        for i in reversed(range(1, len(lst_n_feats))):
            ups.append(
                UpBlock(
                    in1_channels=lst_n_feats[i] + 1,
                    in2_channels=lst_n_feats[i - 1] + 1,
                    out_channels=lst_n_feats[i - 1],
                    bias=bias,
                    upfactor=2,
                    n_layers_in_block=n_layers_in_block,
                    activation_layer=act_layer_maker(activation_type_up),
                )
            )
        self.ups = nn.ModuleList(ups)

        self.last = nn.Conv3d(
            in_channels=lst_n_feats[0] + in_channels + 1,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(
        self, x: torch.Tensor, b: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        #
        x0 = self.up0(x)
        x0 = torch.cat([x0, b], dim=1)

        y = self.conv0(x0)
        y = torch.cat([y, b], dim=1)

        feats, blds = [y], []
        for i, down in enumerate(self.downs):
            y = down(y)
            b = self.down_stride2(b)

            y = torch.cat([y, b], dim=1)

            # vertical size is resized from 5 to 6
            if i == 2:
                _, _, D, H, W = y.shape
                assert D == 5
                y = F.interpolate(y, size=(6, H, W), mode=self.mode)
                b = F.interpolate(b, size=(6, H, W), mode=self.mode)

            feats.append(y)
            blds.append(b)

        _y = feats.pop()
        y = self.latent_layers(_y)
        y = y + _y

        for i, up in enumerate(self.ups):
            b, f = blds.pop(), feats.pop()

            # vertical size is size from 6 to 5.
            if i == 1:
                _, _, D, H, W = y.shape
                assert D == 6
                y = F.interpolate(y, size=(5, H, W), mode=self.mode)
                b = F.interpolate(b, size=(5, H, W), mode=self.mode)

            if i > 0:
                y = torch.cat([y, b], dim=1)
            y = up(y, f)

        assert len(feats) == len(blds) == 0

        y = torch.cat([y, x0], dim=1)  # concat along channel dim

        return self.last(y)
