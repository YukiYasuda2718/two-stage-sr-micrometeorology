import argparse
import datetime
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import dask
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from src.analysis.mssg_helper import HR_SLICE, LR_SLICE, load_all_hr, load_all_lr
from src.data.dl_data_maker import (
    make_coarse_grained_dataarray_with_outside_lr_buildings,
    write_out,
)
from src.data.mssg_bldg_helper import (
    calc_ave_pooling_weights,
    calc_is_out_of_bldg,
    make_resized_lr_tz,
    read_building_height_txt,
)
from src.utils.io_grads import get_mssg_grads_dir

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

dask.config.set(**{"array.slicing.split_large_chunks": True})


ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
DL_DATA_ROOT_DIR = f"{ROOT_DIR}/data/processed/DL_data/v02"
os.makedirs(DL_DATA_ROOT_DIR, exist_ok=True)

CASE_NAME = "tokyo-05m-z20"
CONFIG_NAMES = ["tokyo_20m_no_s2srad", "tokyo_20m_vz_no_s2srad", "tokyo_05m"]
assert CONFIG_NAMES[-1] == "tokyo_05m", "tokyo_05m (e.g., hr data) must be the last."

ROOT_RESUT_DIR = pathlib.Path(f"/data/mssg-results/{CASE_NAME}")

HR_BUILDING_TXT_PATH = f"{ROOT_DIR}/python/data/EleTopoZ_tokyo_05m.txt"
LR_BUILDING_TXT_PATH = f"{ROOT_DIR}/python/data/EleTopoZ_tokyo_20m_no_s2srad.txt"

SR_SCALE = 4
assert SR_SCALE == 4, f"SR_SCALE {SR_SCALE} must be 4."

VAR_NAMES = ["tm", "vl", "vp", "vr"]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--target_datetime",
    type=str,
    help="target UTC datetime in ISO 8601 format, e.g., 2015-07-31T05:00:00",
)

if __name__ == "__main__":
    try:
        target_datetime = datetime.datetime.strptime(
            parser.parse_args().target_datetime, "%Y-%m-%dT%H:%M:%S"
        )
        output_dir_path = f"{DL_DATA_ROOT_DIR}/{datetime.datetime.strftime(target_datetime, '%Y%m%dT%H%M%S')}"
        os.makedirs(output_dir_path, exist_ok=False)

        logger.addHandler(FileHandler(f"{output_dir_path}/log.txt"))

        start_time = time.time()

        logger.info("\n*********************************************************")
        logger.info(f"Start: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")
        logger.info(f"target_datetimes = {target_datetime}")
        logger.info(f"Output dir = {output_dir_path}")

        # Change axes from (x,y) --> (y,x) by transpose
        hr_tz = read_building_height_txt(HR_BUILDING_TXT_PATH, "Tz").transpose()
        hr_ez = read_building_height_txt(HR_BUILDING_TXT_PATH, "Ez").transpose()
        assert (400, 400) == hr_tz.shape == hr_ez.shape
        hr_tz = hr_tz[HR_SLICE["lat"], HR_SLICE["lon"]]
        hr_ez = hr_ez[HR_SLICE["lat"], HR_SLICE["lon"]]

        # Change axes from (x,y) --> (y,x) by transpose
        lr_tz = read_building_height_txt(LR_BUILDING_TXT_PATH, "Tz").transpose()
        lr_ez = read_building_height_txt(LR_BUILDING_TXT_PATH, "Ez").transpose()
        assert (100, 100) == lr_tz.shape == lr_ez.shape
        lr_tz = lr_tz[LR_SLICE["lat"], LR_SLICE["lon"]]
        lr_ez = lr_ez[LR_SLICE["lat"], LR_SLICE["lon"]]

        hr_levs, lr_levs = None, None
        hr_is_out_of_bldg = None
        lr_is_out_of_bldg = None

        for config_name in CONFIG_NAMES:
            logger.info(f"\nconfig name = {config_name}")

            grads_dir = get_mssg_grads_dir(ROOT_RESUT_DIR, target_datetime, config_name)

            all_data = None
            resolution = None
            is_out_of_bldg = None

            if config_name == "tokyo_05m":
                all_data = load_all_hr(grads_dir, VAR_NAMES, extrapolate_nan=True)
                resolution = "hr"

                hr_levs = all_data["tm"].lev.values
                hr_is_out_of_bldg = calc_is_out_of_bldg(
                    tz=hr_tz, ez=hr_ez, actual_levs=hr_levs
                )
                is_out_of_bldg = hr_is_out_of_bldg

            elif config_name == "tokyo_20m_no_s2srad":
                all_data = load_all_lr(grads_dir, VAR_NAMES, extrapolate_nan=True)
                resolution = "lr"

            elif config_name == "tokyo_20m_vz_no_s2srad":
                all_data = load_all_lr(grads_dir, VAR_NAMES, extrapolate_nan=True)
                resolution = "lr"

            else:
                raise ValueError(f"{config_name} is not supported.")

            if resolution == "lr":
                if lr_levs is not None:
                    assert_array_equal(lr_levs, all_data["tm"].lev.values)
                else:
                    lr_levs = all_data["tm"].lev.values
                    lr_is_out_of_bldg = calc_is_out_of_bldg(
                        tz=lr_tz, ez=lr_ez, actual_levs=lr_levs
                    )
                is_out_of_bldg = lr_is_out_of_bldg

            # Check NaN locations
            is_nans = np.isnan(all_data["tm"].isel(time=-1).values)
            is_in_bldg = is_out_of_bldg == 0
            assert np.all(
                is_nans == is_in_bldg
            ), "NaN locations are different from those of bldg data."

            file_header = f"{resolution}_{config_name}_"
            write_out(all_data, output_dir_path, VAR_NAMES, file_header)
            np.save(
                f"{output_dir_path}/{resolution}_is_out_of_bldg.npy", is_out_of_bldg
            )

            # Hereafter, HR data is resized to LR, continue in the case of LR
            if resolution != "hr":
                continue

            assert lr_is_out_of_bldg is not None
            assert hr_is_out_of_bldg is not None

            logger.info("\nResize lr bldg info.")
            resized_lr_out_of_bldg = make_resized_lr_tz(
                lr_is_out_of_build=lr_is_out_of_bldg,
                hr_is_out_of_build=hr_is_out_of_bldg,
            )

            logger.info("\nCalc weights for avg pooling.")
            avg_pooling_weights = calc_ave_pooling_weights(
                lr_is_out_of_build=resized_lr_out_of_bldg,
                lev_window_width=SR_SCALE,
                lat_window_width=SR_SCALE,
                lon_window_width=SR_SCALE,
            )

            logger.info("\nStart to resize HR data.")

            for itime, dt in enumerate(all_data["tm"]["time"]):
                resized_output = []
                for v in VAR_NAMES:
                    resized = make_coarse_grained_dataarray_with_outside_lr_buildings(
                        da=all_data[v].isel(time=itime),
                        org_lr_is_out_of_build=lr_is_out_of_bldg,
                        resized_lr_is_out_of_build=resized_lr_out_of_bldg,
                        hr_is_out_of_build=hr_is_out_of_bldg,
                        avg_pooling_weights=avg_pooling_weights,
                        lev_window_width=SR_SCALE,
                        lat_window_width=SR_SCALE,
                        lon_window_width=SR_SCALE,
                    )
                    resized_output.append(resized.values)

                output = np.stack(resized_output)  # stack along axis = 0
                path = f"{output_dir_path}/lr_{config_name}_{pd.Timestamp(dt.values):%Y%m%dT%H%M%S}.npy"

                np.save(path, output.astype(np.float32))
                logger.info(f"{path} has been generated. Shape = {output.shape}")

        logger.info("\n*********************************************************")
        logger.info(f"Elapsed = {(time.time() - start_time)/60} [min]")
        logger.info(f"End: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
