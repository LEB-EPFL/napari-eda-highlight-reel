from asyncore import write
from napari_eda_highlight_reel._widget import Extractor_Widget
from napari_eda_highlight_reel._writer import write_single_image, write_multiple

from .test_widget import extractor_after_load

import napari

from pathlib import Path

import pytest

# add your tests here...
write_test_path = str(Path(__file__).parents[3] / 'images' / 'images_tests' / 'try.zarr')

def test_zarr_writer(extractor_after_load: Extractor_Widget, make_napari_viewer: napari.Viewer):
    lays_info = [lay.as_layer_data_tuple() for lay in extractor_after_load._viewer.layers]
    write_multiple(write_test_path, lays_info)
    vieww = make_napari_viewer()
    vieww.open(write_test_path + '/Images', plugin='napari-ome-zarr')
    for lay in vieww.layers:
        assert lay.data.shape == extractor_after_load._viewer.layers[lay.name].data.shape