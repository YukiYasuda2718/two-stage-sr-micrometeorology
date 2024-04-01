from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from src.utils.io_grads import (
    align_nan_grids,
    read_xarray_dataarray,
    set_nan_and_extrapolate_nan,
)

logger = getLogger()

HR_SLICE = {
    "time": slice(0, None),
    "lev": slice(2, 42),
    "lat": slice(40, -40),
    "lon": slice(40, -40),
}
LR_SLICE = {
    "time": slice(0, None),
    "lev": slice(1, 11),
    "lat": slice(10, -10),
    "lon": slice(10, -10),
}

HR_SLICE_SJK = {
    "time": slice(0, None),
    "lev": slice(6, 46),
    "lat": slice(40, -40),
    "lon": slice(40, -40),
}
LR_SLICE_SJK = {
    "time": slice(0, None),
    "lev": slice(2, 12),
    "lat": slice(10, -10),
    "lon": slice(10, -10),
}


def slice_hr_and_lr_data(
    hr_data: xr.DataArray,
    lr_data: xr.DataArray,
    hr_slice: dict[str, slice] = HR_SLICE,
    lr_slice: dict[str, slice] = LR_SLICE,
):
    return hr_data.isel(hr_slice), lr_data.isel(lr_slice)


def slice_hr_data(data: xr.DataArray, hr_slice: dict[str, slice] = HR_SLICE):
    return data.isel(hr_slice)


def slice_lr_data(data: xr.DataArray, lr_slice: dict[str, slice] = LR_SLICE):
    return data.isel(lr_slice)


def load_all_hr(
    dir_path: str,
    variable_names: list[str],
    extrapolate_nan: bool,
    hr_slice: dict[str, slice] = HR_SLICE,
):
    assert "tm" in variable_names

    all_data = {v: read_xarray_dataarray(dir_path, v) for v in variable_names}

    for v in variable_names:
        all_data[v] = slice_hr_data(all_data[v], hr_slice)

    for v in variable_names:
        if extrapolate_nan and (v == "vl" or v == "vp"):
            all_data[v] = set_nan_and_extrapolate_nan(all_data[v])

        if v != "tm":
            all_data[v] = align_nan_grids(all_data[v], all_data["tm"])

    return all_data


def load_all_lr(
    dir_path: str,
    variable_names: list[str],
    extrapolate_nan: bool,
    lr_slice: dict[str, slice] = LR_SLICE,
):
    assert "tm" in variable_names

    all_data = {v: read_xarray_dataarray(dir_path, v) for v in variable_names}

    for v in variable_names:
        all_data[v] = slice_lr_data(all_data[v], lr_slice)

    for v in variable_names:
        if extrapolate_nan and (v == "vl" or v == "vp"):
            all_data[v] = set_nan_and_extrapolate_nan(all_data[v])

        if v != "tm":
            all_data[v] = align_nan_grids(all_data[v], all_data["tm"])

    return all_data


def apply_avg_pooling_3d(data: np.ndarray, kernel_size: int = 4) -> np.ndarray:
    assert isinstance(data, np.ndarray)
    assert data.ndim == 4  # time, z, y, x
    assert data.shape[-1] % kernel_size == 0
    assert data.shape[-2] % kernel_size == 0
    assert data.shape[-3] % kernel_size == 0

    return (
        F.avg_pool3d(torch.from_numpy(data)[None, ...], kernel_size=kernel_size)
        .squeeze()
        .numpy()
    )


def calc_correlations_as_2d_images(
    hr_data: xr.DataArray,
    lr_data: xr.DataArray,
):
    hr = (
        F.avg_pool3d(torch.from_numpy(hr_data.values)[None, ...], kernel_size=4)
        .squeeze()
        .numpy()
    )

    lr = lr_data.values

    assert hr.shape == lr.shape, f"{hr.shape}, {lr.shape}"

    hr = hr - np.nanmean(hr, axis=(-2, -1), keepdims=True)
    hr = hr / np.sqrt(np.nanmean(hr**2, axis=(-2, -1), keepdims=True))

    lr = lr - np.nanmean(lr, axis=(-2, -1), keepdims=True)
    lr = lr / np.sqrt(np.nanmean(lr**2, axis=(-2, -1), keepdims=True))

    return np.nanmean(hr * lr, axis=(-2, -1))


def calc_differences_as_2d_images(
    hr_data: xr.DataArray,
    lr_data: xr.DataArray,
):
    hr = (
        F.avg_pool3d(torch.from_numpy(hr_data.values)[None, ...], kernel_size=4)
        .squeeze()
        .numpy()
    )

    lr = lr_data.values

    assert hr.shape == lr.shape, f"{hr.shape}, {lr.shape}"

    abs_diffs = np.abs(hr - lr)

    return np.nanmean(abs_diffs, axis=(-2, -1))


def calc_cross_correlations_as_2d_images(
    hr_data: xr.DataArray,
    lr_data: xr.DataArray,
    max_lag: int,
):
    hr = (
        F.avg_pool3d(torch.from_numpy(hr_data.values)[None, ...], kernel_size=4)
        .squeeze()
        .numpy()
    )

    lr = lr_data.values

    assert hr.shape == lr.shape, f"{hr.shape}, {lr.shape}"

    hr = hr - np.nanmean(hr, axis=(0, -2, -1), keepdims=True)
    hr = hr / np.sqrt(np.nanmean(hr**2, axis=(0, -2, -1), keepdims=True))

    lr = lr - np.nanmean(lr, axis=(0, -2, -1), keepdims=True)
    lr = lr / np.sqrt(np.nanmean(lr**2, axis=(0, -2, -1), keepdims=True))

    _shape = (max_lag + 1, lr.shape[1])  # lag and vertical
    C = np.zeros(_shape)

    for lag in range(0, max_lag + 1, 1):
        assert lag >= 0

        if lag > 0:
            X0 = hr[:-lag, ...]
            X1 = lr[lag:, ...]
        else:
            X0 = hr
            X1 = lr
        assert X0.shape == X1.shape

        C[lag] = np.nanmean(X0 * X1, axis=(0, -2, -1))  # vertical dim remains.

    return C


def calc_correlations_as_2d_images_without_resize(data1: np.ndarray, data2: np.ndarray):
    assert data1.shape == data2.shape

    d1 = data1
    d2 = data2

    d1 = d1 - np.nanmean(d1, axis=(-2, -1), keepdims=True)
    d1 = d1 / np.sqrt(np.nanmean(d1**2, axis=(-2, -1), keepdims=True))

    d2 = d2 - np.nanmean(d2, axis=(-2, -1), keepdims=True)
    d2 = d2 / np.sqrt(np.nanmean(d2**2, axis=(-2, -1), keepdims=True))

    return np.nanmean(d1 * d2, axis=(-2, -1))


def calc_differences_as_2d_images_without_resize(data1: np.ndarray, data2: np.ndarray):
    assert data1.shape == data2.shape

    abs_diffs = np.abs(data1 - data2)

    return np.nanmean(abs_diffs, axis=(-2, -1))
