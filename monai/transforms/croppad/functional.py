from typing import Optional, Sequence, Union

import torch
from monai.data import get_track_meta

from monai.utils import GridSamplePadMode, NumpyPadMode, convert_to_tensor


def croppad(
        img: torch.Tensor,
        slices: Union[Sequence[slice], slice],
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        shape_override: Optional[Sequence] = None
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    input_shape = img_.shape if shape_override is None else shape_override
    input_ndim = len(input_shape) - 1
    if len(slices) != input_ndim:
        raise ValueError(f"'slices' length {len(slices)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    img_centers = [i / 2 for i in input_shape[1:]]
    slice_centers = [(s.stop + s.start) / 2 for s in slices]
    deltas = [s - i for i, s in zip(img_centers, slice_centers)]
    transform = MatrixFactory.from_tensor(img).translate(deltas).matrix.matrix
    im_extents = extents_from_shape([input_shape[0]] + [s.stop - s.start for s in slices])
    im_extents = [transform @ e for e in im_extents]
    shape_override_ = shape_from_extents(input_shape, im_extents)

    metadata = {
        "slices": slices,
        "padding_mode": padding_mode,
        "dtype": img.dtype,
        "im_extents": im_extents,
        "shape_override": shape_override_
    }
    return img_, transform, metadata
