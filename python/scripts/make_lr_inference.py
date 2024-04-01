import os
import pathlib
import shutil
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import torch
import yaml
from src.data.dataloader import make_dataloaders_and_samplers
from src.models.model_maker import make_model
from src.utils.random_seed_helper import set_seeds
from tqdm import tqdm

set_seeds(42)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic

# Configuration for LR inference model
EXPERIMENT_NAME = "lr-inference"
CONFIG_NAME = "default_lr"


ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())
DEVICE = "cuda"

CONFIG_PATH = f"{ROOT_DIR}/python/configs/{EXPERIMENT_NAME}/{CONFIG_NAME}.yml"
OUTPUT_DIR = f"{ROOT_DIR}/data/processed/DL_inference/{EXPERIMENT_NAME}/{CONFIG_NAME}"
_dir = f"{ROOT_DIR}/data/models/{EXPERIMENT_NAME}/{CONFIG_NAME}"
WEIGHT_PATH = f"{_dir}/model_weight.pth"

assert os.path.exists(CONFIG_PATH)
assert os.path.exists(WEIGHT_PATH)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=False)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.addHandler(FileHandler(f"{OUTPUT_DIR}/log.txt"))
logger.setLevel(INFO)


if __name__ == "__main__":
    try:
        with open(CONFIG_PATH) as file:
            config = yaml.safe_load(file)
            config["data"]["batch_size"] = 1
            # This must be 1 to easily obtain timestamps.

        model = make_model(config["model"]).to(DEVICE)
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
        _ = model.eval()

        dataloaders, _ = make_dataloaders_and_samplers(
            root_dir=ROOT_DIR, config=config["data"]
        )

        for kind, dataloader in dataloaders.items():
            logger.info(f"\nKind = {kind}")
            logger.info("Making LR inferences.")

            dataset = dataloader.dataset
            x_shape = (4, 10, 80, 80)
            if dataset.n_input_snapshots == 3:
                x_shape = (12, 10, 80, 80)

            wall_times = []

            for idx in tqdm(range(len(dataset))):
                t, x, b, _, path = dataset.__getitem__(idx=idx, return_hr_path=True)
                assert t.shape == (1,)
                assert x.shape == x_shape
                assert b.shape == (1, 10, 80, 80)

                # add batch dim
                t = t[None, ...]
                x = x[None, ...]
                b = b[None, ...]

                start = time.time()

                pred = (
                    model(t=t.to(DEVICE), x=x.to(DEVICE), b=b.to(DEVICE))
                    .squeeze()
                    .detach()
                    .cpu()
                )

                # Delete batch dim of `b`, and insert NaNs inside bldgs.
                b = torch.broadcast_to(b.squeeze(0), pred.shape)
                pred = torch.where(
                    b == 1.0, pred, torch.full_like(pred, fill_value=torch.nan)
                )

                # Dimensionalize
                pred = dataset._scale_inversely(pred)
                pred = pred.numpy().astype(np.float32)

                assert pred.shape == (4, 10, 80, 80)

                end = time.time()
                wall_times.append(end - start)

                # e.g., lr_tokyo_05m_20130709T040100.npy --> 20130709T040100.npy
                out_path = os.path.basename(path).split("_")[-1]
                out_path = f"{OUTPUT_DIR}/{out_path}"

                assert not os.path.exists(out_path)
                np.save(out_path, pred)

            logger.info(
                f"End: total inference time {np.sum(wall_times)} [sec] to make {len(wall_times)} snapshots"
            )

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
