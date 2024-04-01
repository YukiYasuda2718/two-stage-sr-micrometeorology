from logging import getLogger

import torch

logger = getLogger()


class RandomCrop3D:
    """
    https://discuss.pytorch.org/t/efficient-way-to-crop-3d-image-in-pytorch/78421
    """

    def __init__(self, img_sz: list[int], crop_sz: list[int]):
        assert img_sz[0] >= crop_sz[0]
        assert img_sz[1] >= crop_sz[1]
        assert img_sz[2] >= crop_sz[2]
        self.img_sz = tuple(img_sz)
        self.crop_sz = tuple(crop_sz)

    def __call__(self, x: torch.Tensor):
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return self._crop(x, *slice_hwd)

    @staticmethod
    def _get_slice(sz: int, crop_sz: int):
        if sz == crop_sz:
            lower_bound = 0
        else:
            lower_bound = torch.randint(sz - crop_sz + 1, (1,)).item()
            # max (sz - crop_sz) is exclusive, so + 1 is needed.
        return lower_bound, lower_bound + crop_sz

    @staticmethod
    def _crop(x, slice_h: tuple[int], slice_w: tuple[int], slice_d: tuple[int]):
        logger.debug(f"slice_h = {slice_h}, slice_w = {slice_w}, slice_d = {slice_d}")
        return x[
            ...,
            slice_h[0] : slice_h[1],
            slice_w[0] : slice_w[1],
            slice_d[0] : slice_d[1],
        ]
