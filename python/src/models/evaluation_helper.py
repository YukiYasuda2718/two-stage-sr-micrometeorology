import sys
import typing
from logging import getLogger

import numpy as np
import torch
from src.models.ssim import SSIM3D
from torch import nn
from torch.utils.data import DataLoader

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def evaluate(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fns: typing.Dict[str, typing.Callable],
    device: str,
    hide_progress_bar: bool = False,
    num_evaluation_loops: int = 1,
) -> dict[str, torch.Tensor]:
    #
    _ = model.eval()
    dict_loss = {k: [] for k in loss_fns.keys()}
    tot = num_evaluation_loops * len(dataloader)

    with torch.no_grad(), tqdm(total=tot, disable=hide_progress_bar) as t:
        for n in range(num_evaluation_loops):
            t.set_description(f"Loop {n+1}/{num_evaluation_loops}")

            for ts, Xs, bs, ys in dataloader:
                ts, Xs, bs, ys = (
                    ts.to(device),
                    Xs.to(device),
                    bs.to(device),
                    ys.to(device),
                )
                preds = model(t=ts, x=Xs, b=bs)

                for loss_name, loss_fn in loss_fns.items():
                    lss = loss_fn(predicts=preds, targets=ys, masks=bs).detach().cpu()
                    assert lss.ndim == 2  # dims: batch, z
                    assert lss.shape[0] == ys.shape[0]  # check batch dim
                    assert lss.shape[1] == ys.shape[2]  # check z dim
                    dict_loss[loss_name].append(lss)
                t.update(1)

    for k in dict_loss.keys():
        dict_loss[k] = torch.cat(dict_loss[k], dim=0)
        # concat along bach dim

    return dict_loss


class TemperatureErrorNorm(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert (
            predicts.ndim == targets.ndim == masks.ndim == 5
        )  # dims: batch, channel, z, y, x

        # Temperature is the first channel
        abs_diffs = torch.abs(predicts[:, 0] - targets[:, 0]) * self.scale

        # Sum along y and x dims
        sum_diff = torch.sum(abs_diffs * masks[:, 0], dim=(-2, -1))
        normalize = torch.sum(masks[:, 0], dim=(-2, -1))
        assert (
            sum_diff.shape == normalize.shape == (predicts.shape[0], predicts.shape[2])
        )

        return sum_diff / normalize


class TemperatureErrorNormNormalized(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert (
            predicts.ndim == targets.ndim == masks.ndim == 5
        )  # dims: batch, channel, z, y, x

        # Temperature is the first channel
        abs_diffs = torch.abs(predicts[:, 0] - targets[:, 0]) * self.scale

        # Sum along y and x dims
        sum_diff = torch.sum(abs_diffs * masks[:, 0], dim=(-2, -1))
        normalize = torch.sum(
            masks[:, 0] * torch.abs(targets[:, 0] * self.scale), dim=(-2, -1)
        )
        assert (
            sum_diff.shape == normalize.shape == (predicts.shape[0], predicts.shape[2])
        )

        return sum_diff / normalize


class VelocityErrorNorm(nn.Module):
    def __init__(self, scales: float, device: str):
        super().__init__()
        self.scales = torch.tensor(scales, device=device)
        self.scales = self.scales[None, :, None, None, None]

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert predicts.ndim == targets.ndim == masks.ndim == 5

        # Velocity channels from 1 to 3 (channel 0 is temperature)
        diffs = self.scales * (predicts[:, 1:] - targets[:, 1:])
        norms = torch.sqrt(torch.sum(diffs**2, dim=1))  # Euclidean norm

        sum_norm = torch.sum(norms * masks[:, 0], dim=(-2, -1))
        normalize = torch.sum(masks[:, 0], dim=(-2, -1))
        assert (
            sum_norm.shape == normalize.shape == (predicts.shape[0], predicts.shape[2])
        )

        return sum_norm / normalize


class VelocityErrorNormNormalized(nn.Module):
    def __init__(self, scales: float, device: str):
        super().__init__()
        self.scales = torch.tensor(scales, device=device)
        self.scales = self.scales[None, :, None, None, None]

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert predicts.ndim == targets.ndim == masks.ndim == 5

        # Velocity channels from 1 to 3 (channel 0 is temperature)
        diffs = self.scales * (predicts[:, 1:] - targets[:, 1:])
        norms = torch.sqrt(torch.sum(diffs**2, dim=1))  # Euclidean norm

        sum_norm = torch.sum(norms * masks[:, 0], dim=(-2, -1))

        _n = self.scales * targets[:, 1:]
        _n = torch.sum(_n**2, dim=1)
        _n = torch.sqrt(_n)
        normalize = torch.sum(masks[:, 0] * _n, dim=(-2, -1))

        assert (
            sum_norm.shape == normalize.shape == (predicts.shape[0], predicts.shape[2])
        )

        return sum_norm / normalize


class VelocityComponentErrorNorm(nn.Module):
    def __init__(self, scale: float, idx_channel: int):
        super().__init__()
        self.scale = scale
        self.idx = idx_channel

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert (
            predicts.ndim == targets.ndim == masks.ndim == 5
        )  # dims: batch, channel, z, y, x

        abs_diffs = torch.abs(predicts[:, self.idx] - targets[:, self.idx]) * self.scale

        # Sum along y and x dims
        sum_diff = torch.sum(abs_diffs * masks[:, 0], dim=(-2, -1))
        normalize = torch.sum(masks[:, 0], dim=(-2, -1))
        assert (
            sum_diff.shape == normalize.shape == (predicts.shape[0], predicts.shape[2])
        )

        return sum_diff / normalize


class VelocityComponentErrorNormNormalized(nn.Module):
    def __init__(self, scale: float, idx_channel: int):
        super().__init__()
        self.scale = scale
        self.idx = idx_channel

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert (
            predicts.ndim == targets.ndim == masks.ndim == 5
        )  # dims: batch, channel, z, y, x

        abs_diffs = torch.abs(predicts[:, self.idx] - targets[:, self.idx]) * self.scale

        # Sum along y and x dims
        sum_diff = torch.sum(abs_diffs * masks[:, 0], dim=(-2, -1))
        normalize = torch.sum(
            masks[:, 0] * torch.abs(targets[:, self.idx] * self.scale), dim=(-2, -1)
        )
        assert (
            sum_diff.shape == normalize.shape == (predicts.shape[0], predicts.shape[2])
        )

        return sum_diff / normalize


class AveSsimLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        max_val: float = 1.0,
        eps: float = 1e-7,
        use_gaussian=True,
    ):
        super().__init__()
        self.ssim = SSIM3D(
            window_size=window_size,
            sigma=sigma,
            size_average=False,
            max_val=max_val,
            eps=eps,
            use_gaussian=use_gaussian,
        )

    def forward(
        self, predicts: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ):
        assert (
            predicts.ndim == targets.ndim == masks.ndim == 5
        )  # dims: batch, channel, z, y, x
        _masks = torch.broadcast_to(masks, predicts.shape)

        ssims = self.ssim(predicts, targets, _masks)

        # mean along channel, y and x
        return 1 - torch.mean(ssims, dim=(1, 3, 4))
