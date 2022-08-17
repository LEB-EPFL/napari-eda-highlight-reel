import os

from pathlib import Path

import numpy as np

import napari_eda_highlight_reel._widget as reel
from napari_eda_highlight_reel._widget import Extractor_Widget, Cropper_Widget

import napari

import pytest




# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams

test_image_path= str(Path(__file__).parents[3] / 'images' / 'steven_192.ome.zarr' /'Images')


########## FIXTURES ##########

@pytest.fixture
def extractor_empty(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer(show=False)
    yield Extractor_Widget(viewer)

@pytest.fixture
def extractor_after_load(make_napari_viewer):  #creates Extractor_Widget for the viewer loads data then return the widget
    viewer: napari.Viewer = make_napari_viewer(show=False)
    viewer.open(test_image_path)
    yield Extractor_Widget(viewer)

@pytest.fixture
def cropper_first_event_image(make_napari_viewer):
    viewer: napari.Viewer = make_napari_viewer(show=False)
    viewer.open(test_image_path)
    extractor = Extractor_Widget(viewer)
    extractor.thresh_scroller.setValue(80)
    extractor.full_scan()
    yield extractor.event_list.itemWidget(extractor.event_list.item(0))

########## TESTS ##########

############ Test Extractor Widget ################

def test_Extractor_labels(extractor_empty: Extractor_Widget):
    assert extractor_empty.layout().itemAt(0).layout().itemAt(0).widget().text() == 'NN image layer'
    assert extractor_empty.layout().itemAt(1).layout().itemAt(0).widget().text() == 'Neighbourhood Size'
    assert extractor_empty.layout().itemAt(2).layout().itemAt(0).widget().text() == 'Threshold'
    assert extractor_empty.layout().itemAt(3).layout().itemAt(0).widget().text() == 'Scan'
    assert extractor_empty.layout().itemAt(3).layout().itemAt(1).widget().text() == 'Add'
    assert extractor_empty.layout().itemAt(5).layout().itemAt(0).widget().text() == 'Save all'
    assert extractor_empty.layout().itemAt(5).layout().itemAt(1).widget().text() == 'View all'

def test_EDA_chosen(extractor_after_load: Extractor_Widget):
    assert extractor_after_load.eda_layer.name == 'NN Image'

def test_source(extractor_after_load: Extractor_Widget):
    assert extractor_after_load.image_path == test_image_path

def test_basic_scan(extractor_after_load: Extractor_Widget):
    expected_events = 2
    extractor_after_load.thresh_scroller.setValue(80)
    events = extractor_after_load.basic_scan(extractor_after_load.eda_layer)
    assert len(events) == expected_events

def test_add_event(extractor_after_load: Extractor_Widget):
    extractor_after_load.create_new_event()
    assert extractor_after_load.event_list.count() == 1

def test_create_layer_label(extractor_after_load: Extractor_Widget):
    extractor_after_load.update_event_labels()
    assert extractor_after_load._viewer.layers.__contains__('Event Labels')
    assert extractor_after_load._viewer.layers['Event Labels'].data.shape == extractor_after_load.eda_layer.data.shape

def test_update_layer_label(extractor_after_load: Extractor_Widget):
    extractor_after_load.thresh_scroller.setValue(77)
    extractor_after_load.update_event_labels()
    assert extractor_after_load._viewer.layers['Event Labels'].data[5,0,128,128] == 0
    extractor_after_load.full_scan()
    extractor_after_load.update_event_labels()
    assert extractor_after_load._viewer.layers['Event Labels'].data[5,0,128,128] == 1
    extractor_after_load.event_list.clear()
    extractor_after_load.update_event_labels()
    assert extractor_after_load._viewer.layers['Event Labels'].data[5,0,128,128] == 0
    


############ Cropper Widget ################

def test_Cropper_labels(cropper_first_event_image: Cropper_Widget):
    assert cropper_first_event_image.layout().itemAt(0).layout().itemAt(0).widget().text() == '1'
    assert cropper_first_event_image.layout().itemAt(0).layout().itemAt(1).widget().text() == 'Name:'
    assert cropper_first_event_image.layout().itemAt(0).layout().itemAt(3).widget().text() == 'Layers'
    assert cropper_first_event_image.layout().itemAt(0).layout().itemAt(4).widget().text() == 'X'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(0,1).widget().text() == 'In Space:'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(1,1).widget().text() == 'Center'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(1,2).widget().text() == 'Size'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(2,0).widget().text() == 'x'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(3,0).widget().text() == 'y'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(4,0).widget().text() == 'z'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(0,3).widget().text() == 'Frames:'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(1,3).widget().text() == 'First'
    assert cropper_first_event_image.layout().itemAt(1).layout().itemAtPosition(3,3).widget().text() == 'Last'
    assert cropper_first_event_image.layout().itemAt(2).layout().itemAt(0).widget().text() == 'View'
    assert cropper_first_event_image.layout().itemAt(2).layout().itemAt(1).widget().text() == 'Save'







############ Test Cropper Auxiliary ###############

def test_layer_crop_central():
    dims = [50,50,50,50]
    image = np.random.rand(dims[0],dims[1],dims[2],dims[3])
    limits = []
    sizes = []
    for i in range(4):
        limits.append(sorted([np.random.randint(10,dims[i]-10),np.random.randint(10,dims[i]-10)]))
        sizes.append(limits[i][1]-limits[i][0])
        if sizes[-1] == 0:
            sizes[-1] = 1
    newimg = reel.layer_crop(image,limits)
    assert newimg.shape == tuple(sizes)

def test_layer_crop_border():
    dims = [50,50,50,50]
    image = np.random.rand(dims[0],dims[1],dims[2],dims[3])
    limits = []
    sizes = []
    for i in range(4):
        limits.append(sorted([np.random.randint(10,dims[i]-10),np.random.randint(dims[i],dims[i]+20)]))
        sizes.append(dims[i]-limits[i][0])
    newimg = reel.layer_crop(image,limits)
    assert newimg.shape == tuple(sizes)


############## Test Extractor Auxiliary ##################

def test_find_cool_thing_in_frame():
    dims = 50, 50, 50
    map = np.zeros(dims)
    rng = np.random.default_rng()
    maxnum = np.random.randint(1,10)
    xpos = rng.choice(50, size = maxnum, replace = False)
    ypos = rng.choice(50, size = maxnum, replace = False)
    zpos = rng.choice(50, size = maxnum, replace = False)
    for i in range(maxnum):
        map[xpos[i],ypos[i],zpos[i]] = 1
    ev_c = reel.find_cool_thing_in_frame(map, 0.5, 1)
    assert len(ev_c) == maxnum

############ Test Extractor ################
