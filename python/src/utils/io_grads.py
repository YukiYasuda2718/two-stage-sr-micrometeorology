import datetime
import pathlib
from logging import getLogger

import numpy as np
import xarray as xr
from xgrads import open_CtlDataset

logger = getLogger()


def get_mssg_grads_dir(
    root_dir: pathlib.Path, dt: datetime.datetime, config_name: str
) -> pathlib.Path:
    str_dt = f"{dt:%Y-%m-%dT%H:%M:%S}"
    return root_dir / str_dt / config_name / "mssg" / "out" / "grads"


def align_nan_grids(target_da: xr.DataArray, source_da: xr.DataArray) -> xr.DataArray:
    ret = xr.where(np.isnan(source_da), np.nan, target_da)
    ret.name = target_da.name
    return ret


def read_xarray_dataarray(
    dir_path: pathlib.Path,
    variable_name: str,
    nest_level: str = "0n",
) -> xr.DataArray:
    ds = open_CtlDataset(str(dir_path / f"atmos_{nest_level}_{variable_name}.ctl"))
    da = ds[variable_name]

    # Fill missing values with nan
    assert isinstance(ds.undef, float)
    logger.debug(f"{variable_name}: undef = {ds.undef}")
    da = xr.where(da == ds.undef, np.nan, da)

    return da


def set_nan_and_extrapolate_nan(
    org_data: xr.DataArray, method: str = "nearest"
) -> xr.DataArray:
    data = xr.where(org_data == 0.0, np.nan, org_data)

    data = data.chunk(dict(lev=-1)).interpolate_na(
        dim="lev", method=method, fill_value="extrapolate"
    )

    if np.sum(np.isnan(data.values)) == 0:
        return data

    logger.info("interpolation along lat.")
    data = data.interpolate_na(dim="lat", method=method, fill_value="extrapolate")

    if np.sum(np.isnan(data.values)) == 0:
        return data

    logger.info("interpolation along lon.")
    data = data.interpolate_na(dim="lon", method=method, fill_value="extrapolate")

    assert np.sum(np.isnan(data.values)) == 0

    return data
