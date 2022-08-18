from ome_zarr import io

import napari_ome_zarr

from os import PathLike
from typing import Optional


def napari_get_reader(path: PathLike) -> Optional[function]:
    paths = [str(path) + '/Images', str(path) + 'EDA']
    right = True
    for pt in paths:
        if not zarr.parse_url()