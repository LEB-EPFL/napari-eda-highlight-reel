"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from signal import signal
from typing import TYPE_CHECKING

from magicgui import magic_factory
import numpy as np
from pandas import array
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QWidget, QScrollBar, QScrollArea
from qtpy.QtCore import Qt, QTimer
import dask.array

from pathlib import Path
import scipy.ndimage as ndi

import napari

class Extractor_Widget(QWidget):
    def __init__(self, napari_viewer, nbh_size = 10):
        super().__init__()
        self._viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.nbh_size = nbh_size
        self.time_data = None
        self.image_path = None

        self.events = []
        self.eda_layer = None
        self.eda_ready = False
        self.max_ev_score = None
        self.threshold = 0

        self.create_threshold_scroller()

        self.create_top_buttons()

        self.scroll_area = QScrollArea()
        self.event_list = QVBoxLayout()
        self.scroll_area.setLayout(self.event_list)

        self.create_bottom_buttons()
        
        self.layout().addLayout(self.thresh_grid)
        self.layout().addLayout(self.top_btn_layout)
        self.layout().addWidget(self.scroll_area)
        self.layout().addLayout(self.bottom_btn_layout)

        self.init_data()

        self.scan_btn.clicked.connect(self.full_scan)

        

    def create_threshold_scroller(self):
        self.thresh_grid = QGridLayout()
        self.thresh_scroller= QScrollBar(Qt.Horizontal)
        self.thresh_scroller.setMinimum(0)
        self.thresh_scroller.setSingleStep(1)
        self.thresh_scroller.setMaximum(100)
        self.thresh_scroller.setMinimumWidth(150)
        self.thresh_grid.addWidget(QLabel('Threshold'),0,0)
        self.thresh_show = QLabel('-')
        self.thresh_grid.addWidget(self.thresh_show,0,1)
        self.thresh_grid.addWidget(self.thresh_scroller,1,0,1,2)
    
    def create_top_buttons(self):
        self.scan_btn = QPushButton('Scan')
        self.add_btn = QPushButton('Add')
        self.top_btn_layout = QHBoxLayout()
        self.top_btn_layout.addWidget(self.scan_btn)
        self.top_btn_layout.addWidget(self.add_btn)
        

    def create_bottom_buttons(self):
        self.save_all_btn = QPushButton('Save all')
        self.view_all_btn = QPushButton('View all')
        self.bottom_btn_layout = QHBoxLayout()
        self.bottom_btn_layout.addWidget(self.save_all_btn)
        self.bottom_btn_layout.addWidget(self.view_all_btn)

    def init_data(self):
        if self.image_path != self._viewer.layers[0].source.path : #update data if new source is added
            self.image_path = self._viewer.layers[0].source.path
        #self.time_data=get_times(self)#init times of initial image
        connect_eda(self)
        self.search_eda_layer()
        if self.eda_ready:
            self.set_max_thresh()
            self.update_threshhold()
            self.thresh_scroller.valueChanged.connect(self.update_threshhold)

    def set_max_thresh(self):
        if type(self.eda_layer.data) == dask.array.core.Array:
            self.max_ev_score = np.amax(self.eda_layer.data.compute())
        else:
            self.max_ev_score = np.amax(np.asarray(self.eda_layer.data))
        

    def search_eda_layer(self):
        self.eda_ready = False
        for lay in self._viewer.layers:
            if lay.name == 'EDA':
                self.eda_layer = lay
                self.eda_ready = True
        if not self.eda_ready:
            self.ask_eda_layer_name()

    def ask_eda_layer_name(self):
        return 0


    def full_scan(self):
        self.clear_event_list()
        if self.eda_ready:
            self.events = self.basic_scan(self.eda_layer)
            for ev in self.events:
                self.event_list.addWidget(Cropper_Widget(self._viewer, ev))
    

    def basic_scan(self,layer):
        open_events = []
        framenumber = len(np.asarray(layer.data))
        for i in range(framenumber):
            actualist = find_cool_thing_in_frame(np.asarray(layer.data)[i],threshold = self.threshold, nbh_size = self.nbh_size)
            while actualist:
                new_event = True
                for ev in open_events:
                    if abs(ev.c_x - actualist[0]['x'])<self.nbh_size and abs(ev.c_y - actualist[0]['y'])<self.nbh_size and abs(ev.c_z - actualist[0]['z'])<self.nbh_size and ev.last_frame == i-1:
                        ev.last_frame = i
                        new_event = False
                if new_event:
                    open_events.append(Event([actualist[0]['x'],actualist[0]['y'],actualist[0]['z']],i))
                actualist.pop(0)
            print('frame number ' + str(i) + 'scanned')
        return open_events
                    
                    
    def clear_event_list(self, L = False):
        if not L:
            L = self.event_list
        if L is not None:
            while L.count():
                item = L.takeAt(0)
                
                widget = item.widget()
                
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearvbox(item.layout())
    
    def update_threshhold(self):
        self.threshold = self.thresh_scroller.value()*self.max_ev_score/100
        self.thresh_show.setText(str(self.threshold))



def find_cool_thing_in_frame(frame, threshold, nbh_size):
    data_max = ndi.filters.maximum_filter(frame, nbh_size, mode = 'constant', cval = 0)
    maxima = (frame == data_max)
    upper = (frame > threshold)
    maxima[upper == 0] = 0
    labeled, num_objects = ndi.label(maxima)
    slices = ndi.find_objects(labeled)
    Events_centers = []
    for dz,dy,dx in slices:
        evvy = {'x': 0, 'y': 0, 'z': 0}
        evvy['x'] = (dx.start + dx.stop - 1)/2
        evvy['y'] = (dy.start + dy.stop - 1)/2
        evvy['z'] = (dz.start + dz.stop - 1)/2
        Events_centers.append(evvy)
    return Events_centers



class Event():
    def __init__(self,center_position, first_frame):
        self.c_x = center_position[0]
        self.c_y = center_position[1]
        self.c_z = center_position[2]
        self.first_frame = first_frame
        self.last_frame = first_frame

    
    

class Cropper_Widget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer, event):
        super().__init__()
        self._viewer = napari_viewer
        self._event = event
        self.time_data = None
        self.event_scores = None

        self.size_slider = QScrollBar(Qt.Horizontal)
        self.hister_slider = QScrollBar(Qt.Horizontal)

        self.view_btn = QPushButton('View')
        self.save_btn = QPushButton('Save')
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.view_btn)
        self.btn_layout.addWidget(self.save_btn)
        
                
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.size_slider)
        self.layout().addWidget(self.hister_slider)
        self.layout().addLayout(self.btn_layout)

        self.view_btn.clicked.connect(self.view_reel)
        self.save_btn.clicked.connect(self.save_reel)

    def full_crop(self,layer_numbers,sizes,hf_frame, hl_frame):
        to_use = self.convert_to_easy_format(layer_numbers)
        return centered_crop(to_use, self.center_position,sizes,hf_frame,hl_frame)

    def convert_to_easy_format(self, layer_numbers):
        ready_video = dict()
        for i in layer_numbers:
            
            video = self._viewer.layers[i].data
            if type(video) == dask.array.core.Array:
                video = video.compute()
            video = np.asarray(video)
            video = video.squeeze()
            if video.ndim == 3:
                video = np.expand_dims(video, axis = 1)
            ready_video[self._viewer.layers[i].name] = video
        return ready_video

    def save_reel(self):
        return 0

    def view_reel(self):
        return 0

    


"""
@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
"""
def centered_crop(videos: dict, Center_pos, Sizes, frame_begin, frame_end): #bref, center pos et Sizes seront donnes dans l'ordre z y x
        limits = []
        for i in range(len(Center_pos)):
            frst = Center_pos[i] - int(0.5*Sizes[i])
            lst = Center_pos[i] + int(0.5*Sizes[i])
            limits.append([frst,lst])
        new_lay = dict()
        for key in videos.keys():
            new_lay[key] = layer_crop(videos[key],[frame_begin, frame_end],limits)
        return new_lay

def layer_crop(image: np.array, time_limits, space_limits: list):
        if image.ndim == len(space_limits)+1:
            newimg = image[time_limits[0]:time_limits[1],space_limits[0][0]:space_limits[0][1],space_limits[1][0]:space_limits[1][1],space_limits[2][0]:space_limits[2][1]]
        else:
            raise
        return newimg

def connect_eda(widget: Extractor_Widget):
   edapath = str(Path(widget.image_path).parent / 'EDA')
   widget._viewer.open(edapath, plugin = "napari-ome-zarr")
   widget._viewer.layers[-1].opacity = 0.25
   widget._viewer.layers[-1].name = 'EDA'