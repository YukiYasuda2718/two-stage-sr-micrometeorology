from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def read_building_height_txt(
    building_path: str, target_col: Literal["Ez", "Tz", "Tzl"], margin: int = 0
) -> np.ndarray:
    with open(building_path, "r") as file:
        lines = file.readlines()

    # tz = build height, ez = ground height, both from sea surface
    # tzl == ez, if there is no overhanging
    cols = ["i", "j", "Ez", "Tz", "Tzl"]
    _dict = {}
    for i, line in enumerate(lines[1:]):  # skip header
        splits = list(
            map(lambda s: s.strip(), filter(lambda s: s != "", line.split(" ")))
        )
        _dict[i] = {k: v for k, v in zip(cols, splits)}

    df_topography = pd.DataFrame.from_dict(_dict).T

    for col in cols:
        if col == "i" or col == "j":
            df_topography[col] = df_topography[col].astype(int)
        else:
            df_topography[col] = df_topography[col].astype(float)

    ret = pd.pivot_table(
        data=df_topography[["i", "j", target_col]],
        values=target_col,
        index="i",
        columns="j",
        aggfunc="max",
    ).values

    if margin == 0:
        return ret
    else:
        return ret[margin:-margin, margin:-margin]


def calc_is_out_of_bldg(
    tz: np.ndarray, ez: np.ndarray, actual_levs: np.ndarray
) -> np.ndarray:
    # tz = build height, ez = ground height, both from sea surface

    assert tz.shape == ez.shape
    assert len(tz.shape) == 2  # y and x
    assert actual_levs.ndim == 1  # z

    _shape = actual_levs.shape + tz.shape  # dims = (z, y, x)
    is_out_of_bldg = np.ones(_shape, dtype=np.int32)

    max_lev = np.max(actual_levs)

    for j in range(is_out_of_bldg.shape[1]):  # y dim
        for i in range(is_out_of_bldg.shape[2]):  # x dim
            t, e = tz[j, i], ez[j, i]
            if e > t:
                raise Exception("ground is taller than bldg.")

            if t >= max_lev:
                is_out_of_bldg[:, j, i] = 0
            else:
                idx_top_of_build = (actual_levs <= t).argmin()
                is_out_of_bldg[:idx_top_of_build, j, i] = 0

    return is_out_of_bldg


def make_resized_lr_tz(
    lr_is_out_of_build: np.ndarray,
    hr_is_out_of_build: np.ndarray,
):
    assert lr_is_out_of_build.ndim == hr_is_out_of_build.ndim == 3

    # add batch and channel dims before interpolation
    return (
        F.interpolate(
            torch.from_numpy(lr_is_out_of_build[None, None, ...]).to(torch.float32),
            size=hr_is_out_of_build.shape,
            mode="nearest-exact",
        )
        .squeeze()
        .numpy()
        .astype(int)
    )


def calc_ave_pooling_weights(
    lr_is_out_of_build: np.ndarray,
    lev_window_width: int,
    lat_window_width: int,
    lon_window_width: int,
):
    assert lr_is_out_of_build.ndim == 3  # z, y, x

    weights = np.where(
        lr_is_out_of_build == 0,  # inside
        np.zeros_like(lr_is_out_of_build),
        np.ones_like(lr_is_out_of_build),
    )
    weights = np.lib.stride_tricks.sliding_window_view(
        weights,
        window_shape=(lev_window_width, lat_window_width, lon_window_width),
        axis=(0, 1, 2),
    )

    sum_weights = np.sum(weights, axis=(-3, -2, -1), keepdims=True)
    sum_weights = np.broadcast_to(sum_weights, shape=weights.shape)

    # ignore invalid value encounter due to zero-division
    with np.errstate(invalid="ignore"):
        weights = np.where(sum_weights != 0, weights / sum_weights, np.nan)

    return weights
