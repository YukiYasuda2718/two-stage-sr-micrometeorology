import copy
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from src.models.neural_nets.finite_difference import (
    differentiate_along_x,
    differentiate_along_y,
    differentiate_along_z,
)
from torch import nn

logger = getLogger()


def make_loss(config: dict) -> nn.Module:
    if config["name"] == "DivergenceGradientL2Loss":
        logger.info("DivergenceGradientL2Loss is created")
        return DivergenceGradientL2Loss(**config)
    else:
        raise NotImplementedError(f'{config["name"]} is not supported.')


def calc_mask_near_build_wall(
    is_out_bldg: torch.Tensor, num_filter_applications: int = 1
) -> torch.Tensor:
    #
    assert len(is_out_bldg.shape) == 5  # dims = [batch, channel, z, y, x]

    is_in_build = 1 - is_out_bldg
    n_channels = is_in_build.shape[1]

    weight = torch.ones(
        size=(n_channels, 1, 3, 3, 3),
        dtype=is_in_build.dtype,
        device=is_in_build.device,
    )
    filtered = is_in_build
    for _ in range(num_filter_applications):
        filtered = F.conv3d(filtered, weight, padding=1, groups=n_channels)

    filtered = torch.where(
        filtered > 0, torch.ones_like(filtered), torch.zeros_like(filtered)
    )

    is_near_wall = torch.where(
        filtered * is_out_bldg > 0,
        torch.ones_like(is_out_bldg),
        torch.zeros_like(is_out_bldg),
    )
    is_near_wall.requires_grad = False

    return is_near_wall


def _calc_residual_continuity_eq(
    velocity: torch.Tensor,
    delta_meter: float = 5.0,
    padding: int = 1,
) -> torch.Tensor:
    assert len(velocity.shape) == 5  # batch, channels, z, y, x
    assert velocity.shape[1] == 3  # 3 channels, u, v, w

    dudx = differentiate_along_x(velocity[:, 0:1], delta_meter, padding)
    dvdy = differentiate_along_y(velocity[:, 1:2], delta_meter, padding)
    dwdz = differentiate_along_z(velocity[:, 2:3], delta_meter, padding)

    residual = dudx + dvdy + dwdz

    return residual


class DivergenceGradientL2Loss(nn.Module):
    def __init__(
        self,
        weight_gradient_loss: float,
        weight_divergence_loss: float,
        scales: list[float],
        delta_meter: float,
        **kwargs,
    ):
        super().__init__()
        assert (
            len(scales) == 3
        ), "velocity components have 3. So scales length must be 3."

        assert weight_gradient_loss >= 0.0 and weight_divergence_loss >= 0.0

        self.weight_gradient_loss = weight_gradient_loss
        self.weight_divergence_loss = weight_divergence_loss
        self.scales = copy.deepcopy(scales)
        self.mean_scale = np.mean(scales)
        self.delta_meter = delta_meter

        logger.info(f"weight grad loss = {self.weight_gradient_loss}")
        logger.info(f"weight divergence loss = {self.weight_divergence_loss}")
        logger.info(f"velocity scales = {self.scales}, its mean = {self.mean_scale}")
        logger.info(f"delta meter = {self.delta_meter}")

    def _calc_loss_terms(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #
        # masks is a binary field that takes 1 out of bldg. and 0 in bldg.
        # each input has dims of [batch, channel, z, y, x]
        #
        diff = predicts - targets
        sq_diff = (diff) ** 2
        mse = torch.mean(sq_diff)

        is_near_walls = calc_mask_near_build_wall(masks)
        grd_mask = masks[:, :, 1:-1, 1:-1, 1:-1] * (
            1 - is_near_walls[:, :, 1:-1, 1:-1, 1:-1]
        )
        grd_mask.requires_grad = False

        grd_x = differentiate_along_x(diff, padding=0)
        grd_y = differentiate_along_y(diff, padding=0)
        grd_z = differentiate_along_z(diff, padding=0)

        grd_sum = grd_x**2 + grd_y**2 + grd_z**2

        # multiplication of `4` is necessary becasue predicts and targets channels are 4, but mask channel is 1.
        # + 1 is to avoid zero division.
        grd_mse = torch.sum(grd_sum * grd_mask) / (4 * torch.sum(grd_mask) + 1)

        if self.weight_divergence_loss == 0.0:
            return mse, grd_mse, 0.0

        _scales = torch.tensor(self.scales, device=predicts.device)
        _scales = _scales[None, :, None, None, None]  # batch, channel, z, y, x
        _scales.requires_grad = False

        # The first channel is temperature, so targets[:, 1:] and preds[:, 1:] are used.
        scaled_trgt_v = _scales * targets[:, 1:]
        scaled_pred_v = _scales * predicts[:, 1:]

        trgt_div = _calc_residual_continuity_eq(
            scaled_trgt_v, self.delta_meter, padding=0
        )
        pred_div = _calc_residual_continuity_eq(
            scaled_pred_v, self.delta_meter, padding=0
        )

        diff_div = (
            (trgt_div - pred_div) * self.delta_meter / self.mean_scale
        )  # non-dimensionalized

        # multiplication of `4` is NOT necessary becasue diff_div channel is 1 and mask channel is 1.
        # + 1 is to avoid zero division.
        div_mse = torch.sum((diff_div**2) * grd_mask) / (torch.sum(grd_mask) + 1)

        return mse, grd_mse, div_mse

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        mse, grd_mse, div_mse = self._calc_loss_terms(
            predicts=predicts, targets=targets, masks=masks
        )

        return (
            mse
            + self.weight_gradient_loss * grd_mse
            + self.weight_divergence_loss * div_mse
        )
