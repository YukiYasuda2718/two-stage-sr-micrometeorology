import copy
import datetime
import glob
import os
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from src.utils.random_crop import RandomCrop3D
from torch.utils.data import Dataset

logger = getLogger()


def subsample_hr(
    hr: torch.Tensor, bldg: torch.Tensor, max_z_index: int, obs_ratio: float
) -> torch.Tensor:
    assert hr.ndim == bldg.ndim == 4  # channel, z, y, x
    assert hr.shape[1:] == bldg.shape[1:]

    _, z, y, x = torch.where(bldg[:, :max_z_index] == 1)

    indices = np.arange(len(z))
    indices = np.random.choice(indices, size=int(len(z) * obs_ratio), replace=False)
    z = z[indices]
    y = y[indices]
    x = x[indices]

    obs = torch.full_like(hr, fill_value=torch.nan)
    obs[:, z, y, x] = hr[:, z, y, x]

    return obs


class DatasetLrTUVW(Dataset):
    def __init__(
        self,
        data_dirs: list[str],
        ground_truth_name: str,
        input_name: str,
        num_channels: int,
        sampling_interval_minutes: int,
        biases: list[float],
        scales: list[float],
        use_clipping_ground_truth: bool,
        use_clipping_input: bool,
        missing_value: float,
        clipping_min: float,
        clipping_max: float,
        discarded_minute_range: list[float],
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        logger.info("This data set is for LR inference using DL data v02.")

        assert isinstance(biases, list) and isinstance(scales, list)
        assert len(biases) == len(scales) == num_channels
        assert clipping_max > clipping_min
        assert sampling_interval_minutes >= 0 and isinstance(
            sampling_interval_minutes, int
        )
        assert (
            isinstance(discarded_minute_range, list)
            and len(discarded_minute_range) == 2
            and discarded_minute_range[0] < discarded_minute_range[1]
        )

        self.truth_name = ground_truth_name
        self.input_name = input_name
        self.num_channels = num_channels
        self.sampling_interval_minutes = sampling_interval_minutes
        self.use_clipping_truth = use_clipping_ground_truth
        self.use_clipping_input = use_clipping_input
        self.missing_value = missing_value
        self.clipping_min = clipping_min
        self.clipping_max = clipping_max
        self.discarded_minute_range = copy.deepcopy(discarded_minute_range)
        self.dtype = dtype

        self.n_input_snapshots = 3 if self.sampling_interval_minutes > 0 else 1

        logger.info(f"Truth = {self.truth_name}, Input = {self.input_name}")
        logger.info(f"Discarded minute range = {self.discarded_minute_range}")
        logger.info(f"Num channels = {self.num_channels}")
        logger.info(f"Sampling interval = {self.sampling_interval_minutes} min.")
        logger.info(f"num input snapshots = {self.n_input_snapshots}")
        logger.info(f"Biases = {biases}, Scales = {scales}")
        logger.info(f"Clipping: Truth = {self.use_clipping_truth}")
        logger.info(f"Clipping: Input = {self.use_clipping_input}")
        logger.info(f"missing value = {self.missing_value}")
        logger.info(f"clipping: min {self.clipping_min}, max {self.clipping_max}")

        self.lr_is_out_of_bldg = None
        self._set_bldg_data(data_dirs)
        assert self.lr_is_out_of_bldg is not None, "Bldg data is not set."
        assert self.lr_is_out_of_bldg.shape == (10, 80, 80)

        logger.info("\nSet input file paths")
        self.input_all_file_paths = self._get_file_paths(data_dirs, self.input_name)
        logger.info("\nSet ground truth file paths")
        self.truth_all_file_paths = self._get_file_paths(data_dirs, self.truth_name)
        self._validate_pairs_of_file_paths()
        logger.info("Pairs of file paths have been validated.")

        self.biases = torch.tensor(biases, dtype=self.dtype)
        self.scales = torch.tensor(scales, dtype=self.dtype)

        # Add dims to broadcast when preprocessing.
        self.biases = self.biases[:, None, None, None]
        self.scales = self.scales[:, None, None, None]

    def _set_bldg_data(self, data_dirs: list[str]):
        #
        for dir_path in sorted(data_dirs):
            bldg = np.load(f"{dir_path}/lr_is_out_of_bldg.npy")

            if self.lr_is_out_of_bldg is None:
                self.lr_is_out_of_bldg = bldg
            else:
                assert np.all(bldg == self.lr_is_out_of_bldg), "Bldg data is not unique"

        self.lr_is_out_of_bldg = torch.from_numpy(self.lr_is_out_of_bldg).to(self.dtype)

    def _all_files_exist(
        self, dir_path: int, simulation_name: str, dt: datetime.datetime
    ) -> tuple[bool, list[str]]:
        #
        all_file_paths = []

        lst_minutes = [
            -self.sampling_interval_minutes,
            0,
            self.sampling_interval_minutes,
        ]
        lst_minutes = sorted(set(lst_minutes))

        for minutes in lst_minutes:
            _dt = dt + datetime.timedelta(minutes=minutes)
            file_path = f"{dir_path}/{simulation_name}_{_dt:%Y%m%dT%H%M%S}.npy"

            if not os.path.exists(file_path):
                return False, []

            all_file_paths.append(file_path)

        return True, all_file_paths

    def _get_file_paths(
        self, data_dirs: list[str], simulation_name: str
    ) -> list[list[str]]:
        #
        logger.info(f"Simulation name to get file paths = {simulation_name}")

        all_file_paths = []

        for dir_path in sorted(data_dirs):
            logger.info(f"{dir_path} is used.")
            for file_path in sorted(glob.glob(f"{dir_path}/{simulation_name}_*.npy")):
                #
                if file_path.endswith("_time_series.npy"):
                    continue

                # e.g., lr_tokyo_05m_20130709T040100.npy --> 20130709T040100
                dt = datetime.datetime.strptime(
                    os.path.basename(file_path).split("_")[-1].replace(".npy", ""),
                    "%Y%m%dT%H%M%S",
                )
                if (
                    self.discarded_minute_range[0]
                    <= dt.minute
                    <= self.discarded_minute_range[1]
                ):
                    continue

                all_files_exists, file_paths = self._all_files_exist(
                    os.path.dirname(file_path), simulation_name, dt
                )

                if all_files_exists:
                    all_file_paths.append(file_paths)

        assert len(all_file_paths) > 0

        return all_file_paths

    def _validate_pairs_of_file_paths(self):
        assert len(self.input_all_file_paths) == len(self.truth_all_file_paths)

        for in_paths, tr_paths in zip(
            self.input_all_file_paths, self.truth_all_file_paths
        ):
            assert len(in_paths) == len(tr_paths)

            for p1, p2 in zip(in_paths, tr_paths):
                # Check files' datetimes
                d1 = os.path.basename(p1).split("_")[-1]
                d2 = os.path.basename(p2).split("_")[-1]
                assert d1 == d2, f"Datetime is different. {p1} and {p2}"

    def _read_numpy_data(self, path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).to(self.dtype)

    def _scale_and_clamp(self, data: torch.Tensor, use_clipping: bool) -> torch.Tensor:
        ret = (data - self.biases) / self.scales
        if use_clipping:
            if self.clipping_min is None and self.clipping_max is None:
                pass
            else:
                logger.debug(
                    f"Clipping: min {self.clipping_min}, max {self.clipping_max}"
                )
                ret = torch.clamp(ret, min=self.clipping_min, max=self.clipping_max)
        return ret

    def _scale_inversely(self, data: torch.Tensor) -> torch.Tensor:
        return self.scales * data + self.biases

    def _get_time_minutes(self, file_path: str) -> float:
        logger.debug(f"File path to get time is {file_path}")
        #
        # e.g., lr_tokyo_05m_20130709T040100.npy --> 20130709T040100
        #
        dt = datetime.datetime.strptime(
            os.path.basename(file_path).split("_")[-1].replace(".npy", ""),
            "%Y%m%dT%H%M%S",
        )
        return torch.Tensor([dt.minute]).to(self.dtype)

    def __len__(self):
        return len(self.input_all_file_paths)

    def __getitem__(self, idx: int, return_hr_path: bool = False):
        #
        # ground truth (gt) path
        gt_path = self.truth_all_file_paths[idx][0]
        if self.n_input_snapshots == 3:
            logger.debug(
                f"Central path is used for ground truth from {self.truth_all_file_paths[idx]}"
            )
            gt_path = self.truth_all_file_paths[idx][1]
            # the central path is read from three paths

        time_minutes = self._get_time_minutes(gt_path)
        logger.debug(
            f"Time = {time_minutes} min (base file name = {os.path.basename(gt_path)})"
        )

        gt_data = self._read_numpy_data(gt_path)
        gt_data = self._scale_and_clamp(gt_data, use_clipping=self.use_clipping_truth)
        assert gt_data.shape == (self.num_channels, 10, 80, 80)
        # channel, z, y, and x
        logger.debug(f"GT file path = {'/'.join(gt_path.split('/')[-2:])}")

        assert len(self.input_all_file_paths[idx]) == self.n_input_snapshots

        lr_data = []
        for p in self.input_all_file_paths[idx]:
            lr = self._read_numpy_data(p)
            lr = self._scale_and_clamp(lr, use_clipping=self.use_clipping_input)
            lr_data.append(lr)
            logger.debug(f"LR file path = {'/'.join(p.split('/')[-2:])}")
        lr_data = torch.concat(lr_data, dim=0)
        assert lr_data.shape == (self.num_channels * self.n_input_snapshots, 10, 80, 80)
        # channel, z, y, and x

        logger.debug(f"HR size = {gt_data.shape}, LR size = {lr_data.shape}")
        # dims = (channel, z, y, x)

        lr_data = torch.nan_to_num(lr_data, nan=self.missing_value)
        gt_data = torch.nan_to_num(gt_data, nan=self.missing_value)

        # add batch dim
        bldg = self.lr_is_out_of_bldg.clone()[None, ...]

        if not return_hr_path:
            return time_minutes, lr_data, bldg, gt_data
        else:
            return time_minutes, lr_data, bldg, gt_data, gt_path

    def get_input(self, idx: int):
        raise NotImplementedError()


class DatasetTUVWUsingLrInference(Dataset):
    def __init__(
        self,
        data_dirs: list[str],
        hr_name: str,
        lr_name: str,
        lr_experiment_name: str,
        lr_config_name: str,
        use_lr_inference: bool,
        use_lr_input: bool,
        num_channels: int,
        scale_factor: float,
        biases: list[float],
        scales: list[float],
        use_hr_clipping: bool,
        use_lr_clipping: bool,
        hr_cropped_size: list[int],
        hr_image_size: list[int],
        missing_value: float,
        clipping_min: float,
        clipping_max: float,
        discarded_minute_range: list[float],
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(biases, list) and isinstance(scales, list)
        assert len(biases) == len(scales) == num_channels
        assert clipping_max > clipping_min
        assert (
            isinstance(discarded_minute_range, list)
            and len(discarded_minute_range) == 2
            and discarded_minute_range[0] < discarded_minute_range[1]
        )
        assert (
            hr_cropped_size[0] == hr_image_size[0] == 40
        ), "cropped or image size of z is not 40"

        self.scale_factor = scale_factor
        self.inv_scale_factor = 1 / self.scale_factor

        assert self.scale_factor in [4]
        assert self.inv_scale_factor in [0.25]

        assert use_lr_inference == True or use_lr_input == True

        self.hr_name = hr_name
        self.lr_name = lr_name
        self.lr_experiment_name = lr_experiment_name
        self.lr_config_name = lr_config_name
        self.use_lr_inference = use_lr_inference
        self.use_lr_input = use_lr_input
        self.num_channels = num_channels
        self.hr_clipping = use_hr_clipping
        self.lr_clipping = use_lr_clipping
        self.hr_cropped_size = tuple(hr_cropped_size)  # immutable
        self.hr_image_size = tuple(hr_image_size)
        self.missing_value = missing_value
        self.clipping_min = clipping_min
        self.clipping_max = clipping_max
        self.dtype = dtype

        self.lr_image_size = (
            self.hr_image_size[0] // self.scale_factor,
            self.hr_image_size[1] // self.scale_factor,
            self.hr_image_size[2] // self.scale_factor,
        )

        logger.info(f"HR = {self.hr_name}, LR = {self.lr_name}")
        logger.info(
            f"LR inference from {self.lr_experiment_name}/{self.lr_config_name}"
        )
        logger.info(f"Use LR inference = {self.use_lr_inference}")
        logger.info(f"Use LR input = {self.use_lr_input}")
        logger.info(f"Discarded minute range = {discarded_minute_range}")
        logger.info(f"Num channels = {self.num_channels}")
        logger.info(
            f"Scale factor = {self.scale_factor}, inverse = {self.inv_scale_factor}"
        )
        logger.info(f"Biases = {biases}, Scales = {scales}")
        logger.info(f"Clipping: LR = {self.lr_clipping}, HR = {self.hr_clipping}")
        logger.info(f"3D crop: {self.hr_cropped_size} from {self.hr_image_size}")
        logger.info(f"missing value = {self.missing_value}")
        logger.info(f"clipping: min {self.clipping_min}, max {self.clipping_max}")

        self.hr_is_out_of_bldg = None
        self._set_hr_bldg_data(data_dirs)
        assert self.hr_is_out_of_bldg is not None
        assert self.hr_is_out_of_bldg.shape == (40, 320, 320)  # z, y, x

        logger.info("\nset LR file paths")
        self.lr_all_file_paths = self._get_file_paths(
            data_dirs, self.lr_name, discarded_minute_range
        )

        logger.info("\nset HR file paths")
        self.hr_all_file_paths = self._get_file_paths(
            data_dirs, self.hr_name, discarded_minute_range
        )

        self._extract_paths_having_lr_inferences()
        logger.info("File paths have been made and checked.")

        self.random_3d_crop = RandomCrop3D(self.hr_image_size, self.hr_cropped_size)

        self.biases = torch.tensor(biases, dtype=self.dtype)
        self.scales = torch.tensor(scales, dtype=self.dtype)

        # Add dims to broadcast when preprocessing.
        self.biases = self.biases[:, None, None, None]
        self.scales = self.scales[:, None, None, None]

    def _set_hr_bldg_data(self, data_dirs: list[str]):
        #
        for dir_path in sorted(data_dirs):
            bldg = np.load(f"{dir_path}/hr_is_out_of_bldg.npy")

            if self.hr_is_out_of_bldg is None:
                self.hr_is_out_of_bldg = bldg
            else:
                assert np.all(bldg == self.hr_is_out_of_bldg), "Bldg data is not unique"

        self.hr_is_out_of_bldg = torch.from_numpy(self.hr_is_out_of_bldg).to(self.dtype)

    def _get_file_paths(
        self, data_dirs: list[str], simulation_name: str, discarded_minute_range: list
    ) -> list[str]:
        #
        all_file_paths = []

        for dir_path in sorted(data_dirs):
            logger.info(f"{dir_path} is used.")
            for file_path in sorted(glob.glob(f"{dir_path}/{simulation_name}_*.npy")):
                #
                if file_path.endswith("_time_series.npy"):
                    continue
                #
                # e.g., hr_tokyo_05m_20130709T040100.npy --> 20130709T040100
                #
                dt = datetime.datetime.strptime(
                    os.path.basename(file_path).split("_")[-1].replace(".npy", ""),
                    "%Y%m%dT%H%M%S",
                )
                if discarded_minute_range[0] <= dt.minute <= discarded_minute_range[1]:
                    continue
                all_file_paths.append(file_path)

        assert len(all_file_paths) > 0

        return all_file_paths

    def _extract_paths_having_lr_inferences(self):
        assert len(self.lr_all_file_paths) == len(self.hr_all_file_paths)

        lr_paths = copy.deepcopy(self.lr_all_file_paths)
        hr_paths = copy.deepcopy(self.hr_all_file_paths)

        self.lr_all_file_paths = []
        self.hr_all_file_paths = []
        self.lr_inference_paths = []

        for lr, hr in zip(lr_paths, hr_paths):
            # File names contain datetimes, so this assertion checks the order of datetimes
            # e.g., hr_tokyo_05m_20130709T040100.npy --> 20130709T040100.npy
            assert os.path.basename(lr.split("_")[-1]) == os.path.basename(
                hr.split("_")[-1]
            )

            name = os.path.basename(hr).split("_")[-1]
            inference = hr.split("DL_data")[0]
            inference = f"{inference}DL_inference/{self.lr_experiment_name}/{self.lr_config_name}/{name}"

            if not os.path.exists(inference):
                continue

            self.lr_all_file_paths.append(lr)
            self.hr_all_file_paths.append(hr)
            self.lr_inference_paths.append(inference)

    def _read_numpy_data(self, path: str) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).to(self.dtype)

    def _scale_and_clamp(self, data: torch.Tensor, use_clipping: bool) -> torch.Tensor:
        ret = (data - self.biases) / self.scales
        if use_clipping:
            if self.clipping_min is None and self.clipping_max is None:
                pass
            else:
                logger.debug(
                    f"Clipping: min {self.clipping_min}, max {self.clipping_max}"
                )
                ret = torch.clamp(ret, min=self.clipping_min, max=self.clipping_max)
        return ret

    def _scale_inversely(self, data: torch.Tensor) -> torch.Tensor:
        return self.scales * data + self.biases

    def __len__(self):
        return len(self.lr_all_file_paths)

    def _get_time_minutes(self, file_path: str) -> float:
        logger.debug(f"File path to get time is {file_path}")
        #
        # e.g., hr_tokyo_05m_20130709T040100.npy --> 20130709T040100.npy
        #
        dt = datetime.datetime.strptime(
            os.path.basename(file_path).split("_")[-1].replace(".npy", ""),
            "%Y%m%dT%H%M%S",
        )
        return torch.Tensor([dt.minute]).to(self.dtype)

    def __getitem__(self, idx: int):
        hr_data = self._read_numpy_data(self.hr_all_file_paths[idx])
        lr_data = self._read_numpy_data(self.lr_all_file_paths[idx])
        lr_infr = self._read_numpy_data(self.lr_inference_paths[idx])
        time_minute = self._get_time_minutes(self.hr_all_file_paths[idx])
        # dims = (channel, z, y, x)
        logger.debug(
            f"HR size = {hr_data.shape}, LR size = {lr_data.shape}, LR infr size = {lr_infr.shape}"
        )
        logger.debug(f"Time minutes = {time_minute}")

        logger.debug(
            f"HR size = {hr_data.shape}, LR size = {lr_data.shape}, LR infr size = {lr_infr.shape}"
        )

        assert hr_data.shape[1:] == self.hr_image_size
        assert lr_data.shape[1:] == lr_infr.shape[1:] == self.lr_image_size

        assert (
            hr_data.shape[0]
            == lr_data.shape[0]
            == lr_infr.shape[0]
            == self.num_channels
        )

        hr_data = self._scale_and_clamp(hr_data, use_clipping=self.hr_clipping)
        lr_data = self._scale_and_clamp(lr_data, use_clipping=self.lr_clipping)
        lr_infr = self._scale_and_clamp(lr_infr, use_clipping=self.lr_clipping)

        _data = []
        if self.use_lr_input:
            _data.append(lr_data)
            logger.debug("LR input is concatenated.")

        if self.use_lr_inference:
            _data.append(lr_infr)
            logger.debug("LR inference is concatenated.")

        lr_data = torch.concat(_data, dim=0)
        logger.debug(f"LR shape after concat = {lr_data.shape}")

        # add batch dim, and then interpolate
        lr_data = F.interpolate(
            lr_data.unsqueeze(0), scale_factor=self.scale_factor, mode="nearest-exact"
        ).squeeze()

        hr_bldg = self.hr_is_out_of_bldg.clone()[None, ...]  # add channel dim

        stacked = torch.cat([hr_bldg, hr_data, lr_data], dim=0)
        logger.debug(f"shape of stacked after torch.cat = {stacked.shape}")

        assert stacked.shape[-3:] == self.hr_image_size
        stacked = self.random_3d_crop(stacked)
        logger.debug(f"shape of stacked after cropping = {stacked.shape}")

        hr_bldg = stacked[0:1]
        hr_data = stacked[1 : 1 + self.num_channels]
        lr_data = stacked[1 + self.num_channels :]

        logger.debug(
            f"Shape: bldg = {hr_bldg.shape}, hr = {hr_data.shape}, lr = {lr_data.shape}"
        )

        # add batch dim, and then interpolate
        lr_data = F.interpolate(
            lr_data.unsqueeze(0),
            scale_factor=self.inv_scale_factor,
            mode="nearest-exact",
        ).squeeze()

        logger.debug(f"lr shape after downsample = {lr_data.shape}")

        lr_data = torch.nan_to_num(lr_data, nan=self.missing_value)
        hr_data = torch.nan_to_num(hr_data, nan=self.missing_value)

        return time_minute, lr_data, hr_bldg, hr_data

    def get_input(self, idx: int):
        lr_data = self._read_numpy_data(self.lr_all_file_paths[idx])
        lr_infr = self._read_numpy_data(self.lr_inference_paths[idx])
        hr_path = self.hr_all_file_paths[idx]
        time_minute = self._get_time_minutes(hr_path)
        # dims = (channel, z, y, x)

        lr_data = self._scale_and_clamp(lr_data, use_clipping=self.lr_clipping)
        lr_infr = self._scale_and_clamp(lr_infr, use_clipping=self.lr_clipping)

        _data = []
        if self.use_lr_input:
            _data.append(lr_data)
            logger.debug("LR input is concatenated.")

        if self.use_lr_inference:
            _data.append(lr_infr)
            logger.debug("LR inference is concatenated.")

        lr_data = torch.concat(_data, dim=0)
        logger.debug(f"LR shape after concat = {lr_data.shape}")

        hr_bldg = self.hr_is_out_of_bldg.clone()[None, ...]  # add channel dim

        lr_data = torch.nan_to_num(lr_data, nan=self.missing_value)

        return time_minute, lr_data, hr_bldg, hr_path
