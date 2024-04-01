import copy
import datetime
import pathlib
from collections import OrderedDict
from logging import getLogger

import numpy as np
import pandas as pd
import xarray as xr
from src.utils.io_grads import (
    align_nan_grids,
    read_xarray_dataarray,
    set_nan_and_extrapolate_nan,
)

logger = getLogger()


def make_all_deep_learning_data(
    config_name: str,
    simulation_name: str,
    slices: dict[str, slice],
    timestamp: datetime.datetime,
    target_variables: list[str] = ["tm", "vl", "vp", "vr"],
) -> dict[str, xr.DataArray]:
    #
    dt = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    dir_path = pathlib.Path(f"/data/mssg-results/{config_name}/{dt}")
    dir_path = dir_path / simulation_name / "mssg" / "out" / "grads"

    dataset = OrderedDict()
    assert target_variables[0] == "tm", "The first variable is not tm."

    for v in target_variables:
        assert v in ["tm", "vl", "vp", "vr"], f"{v} is not supported."

        data = read_xarray_dataarray(dir_path, v).isel(slices)

        if v == "vl" or v == "vp":
            data = set_nan_and_extrapolate_nan(data)

        if v != "tm":
            data = align_nan_grids(data, dataset["tm"])

        logger.info(f"{simulation_name}, {v}: shape = {data.shape}")

        dataset[v] = data

    return dataset


def write_out(
    dataset: dict[str, xr.DataArray],
    output_dir: str,
    target_variables: list[str],
    file_header: str = "",
):
    assert "tm" in target_variables, "tm is not contained."

    for dt in dataset["tm"].time:
        output = []
        gt_nans = None

        for v in target_variables:
            assert dataset[v].dims == ("time", "lev", "lat", "lon")

            _data = dataset[v].sel(time=dt).values

            if v == "tm":
                gt_nans = np.isnan(_data)
            else:
                assert np.all(
                    gt_nans == np.isnan(_data)
                ), f"NaN locations are different. v = {v}, t = {dt}"

            output.append(_data)

        output = np.stack(output)  # stack along axis = 0

        output_path = (
            f"{output_dir}/{file_header}{pd.Timestamp(dt.values):%Y%m%dT%H%M%S}.npy"
        )
        np.save(output_path, output.astype(np.float32))
        logger.info(f"{output_path} has been generated. Shape = {output.shape}")


def make_coarse_grained_dataarray_with_outside_lr_buildings(
    *,
    da: xr.DataArray,
    org_lr_is_out_of_build: np.ndarray,
    resized_lr_is_out_of_build: np.ndarray,
    hr_is_out_of_build: np.ndarray,
    avg_pooling_weights: np.ndarray,
    lev_window_width: int,
    lat_window_width: int,
    lon_window_width: int,
):
    assert (
        org_lr_is_out_of_build.ndim == resized_lr_is_out_of_build.ndim == 3
    )  # z, y, x
    assert da.shape == resized_lr_is_out_of_build.shape == hr_is_out_of_build.shape

    hr_data = xr.where(hr_is_out_of_build == 1, da, np.nan)

    hr_data = hr_data.chunk(dict(lev=-1)).interpolate_na(
        dim="lev", method="nearest", fill_value="extrapolate"
    )
    if np.sum(np.isnan(hr_data.values)) > 0:
        hr_data = hr_data.interpolate_na(
            dim="lat", method="nearest", fill_value="extrapolate"
        )
    if np.sum(np.isnan(hr_data.values)) > 0:
        hr_data = hr_data.interpolate_na(
            dim="lon", method="nearest", fill_value="extrapolate"
        )
    hr_data = xr.where(resized_lr_is_out_of_build == 1, hr_data, np.nan)

    _data = np.lib.stride_tricks.sliding_window_view(
        hr_data.values,
        window_shape=(lev_window_width, lat_window_width, lon_window_width),
        axis=(0, 1, 2),
    )
    assert _data.shape == avg_pooling_weights.shape

    _data = np.sum((_data * avg_pooling_weights), axis=(-3, -2, -1))

    lr_data = np.full_like(hr_data.values, np.nan)
    lr_data[
        lev_window_width // 2 : -lev_window_width // 2 + 1,
        lat_window_width // 2 : -lat_window_width // 2 + 1,
        lon_window_width // 2 : -lon_window_width // 2 + 1,
    ] = _data

    lr_da = xr.DataArray(lr_data, coords=copy.deepcopy(da.coords))

    lr_da = lr_da.sel(
        lev=lr_da.lev.values[lev_window_width // 2 :: lev_window_width],
        lat=lr_da.lat.values[lat_window_width // 2 :: lat_window_width],
        lon=lr_da.lon.values[lon_window_width // 2 :: lon_window_width],
    )
    lr_da.coords["lev"] = copy.deepcopy(da.lev[::lev_window_width])
    lr_da.coords["lat"] = copy.deepcopy(da.lat[::lat_window_width])
    lr_da.coords["lon"] = copy.deepcopy(da.lon[::lon_window_width])

    assert lr_da.shape == org_lr_is_out_of_build.shape

    # Check NaN locations
    is_nans = np.isnan(lr_da.values)
    is_in_bldg = org_lr_is_out_of_build == 0
    assert np.all(
        is_nans == is_in_bldg
    ), "NaN locations are different from those of bldg data."

    return lr_da
