import glob
import os
from logging import getLogger

from sklearn.model_selection import train_test_split
from src.data.dataset import DatasetLrTUVW, DatasetTUVWUsingLrInference
from src.utils.random_seed_helper import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


def split_file_paths(
    paths: list[str], train_valid_test_ratios: list[float]
) -> tuple[list[str], list[str], list[str]]:
    logger.info(f"train, valid, test ratios = {train_valid_test_ratios}")

    assert len(train_valid_test_ratios) == 3  # train, valid, test, three ratios
    assert sum(train_valid_test_ratios) == 1.0
    assert all([r > 0 for r in train_valid_test_ratios])

    test_size = train_valid_test_ratios[-1]
    _paths, test_paths = train_test_split(paths, test_size=test_size, shuffle=False)

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_paths, valid_paths = train_test_split(
        _paths, test_size=valid_size, shuffle=False
    )

    assert set(train_paths).isdisjoint(set(valid_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(valid_paths).isdisjoint(set(test_paths))

    logger.info(
        f"train: {len(train_paths)}, valid: {len(valid_paths)}, test: {len(test_paths)}"
    )

    return train_paths, valid_paths, test_paths


def _make_dataloaders_and_samplers(
    *,
    dataset_initilizer,
    dict_dir_paths: dict[str, list[str]],
    train_valid_test_kinds: list[str],
    batch_size: int,
    world_size: int = None,
    rank: int = None,
    num_workers: int = 2,
    seed: int = 42,
    **kwargs,
):
    logger.info(
        f"batch size = {batch_size}, world_size = {world_size}, rank = {rank}, num_workers = {num_workers}, seed = {seed}\n"
    )

    if world_size is not None:
        assert isinstance(rank, int)
        assert batch_size % world_size == 0, "batch_size % world_size /= 0."

    dict_dataloaders, dict_samplers = {}, {}

    for kind in train_valid_test_kinds:
        dataset = dataset_initilizer(data_dirs=dict_dir_paths[kind], **kwargs)

        if world_size is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


def make_dataloaders_and_samplers(
    root_dir: str,
    config: dict,
    train_valid_test_kinds: list[str] = ["train", "valid", "test"],
    world_size: int = None,
    rank: int = None,
):
    dl_data_dir = f"{root_dir}/data/processed/DL_data/{config['dl_data_name']}"

    data_dirs = sorted(
        [path for path in glob.glob(f"{dl_data_dir}/*") if os.path.isdir(path)]
    )

    train_dirs, valid_dirs, test_dirs = split_file_paths(
        data_dirs, config["train_valid_test_ratios"]
    )
    dict_data_dirs = {"train": train_dirs, "valid": valid_dirs, "test": test_dirs}

    if config["dataset_name"] == "DatasetLrTUVW":
        dataset_initilizer = DatasetLrTUVW
        logger.info("Dataset is DatasetLrTUVW")
    elif config["dataset_name"] == "DatasetTUVWUsingLrInference":
        dataset_initilizer = DatasetTUVWUsingLrInference
        logger.info("Dataset is DatasetTUVWUsingLrInference")
    else:
        raise NotImplementedError(f'{config["dataset_name"]} is not supported.')

    return _make_dataloaders_and_samplers(
        dataset_initilizer=dataset_initilizer,
        dict_dir_paths=dict_data_dirs,
        train_valid_test_kinds=train_valid_test_kinds,
        world_size=world_size,
        rank=rank,
        **config,
    )
