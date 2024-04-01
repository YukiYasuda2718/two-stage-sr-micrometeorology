from logging import getLogger

from src.models.neural_nets.unet_hr import UNetHr
from src.models.neural_nets.unet_lr import UNetLr
from torch import nn

logger = getLogger()


def make_model(config: dict) -> nn.Module:
    if config["name"] == "UNetLr":
        logger.info("UNetLr (UNet 1) is created.")
        return UNetLr(**config)
    elif config["name"] == "UNetHr":
        logger.info("UNetHr (UNet 2) is created.")
        return UNetHr(**config)
    else:
        raise NotImplementedError(f'{config["name"]} is not supported.')
