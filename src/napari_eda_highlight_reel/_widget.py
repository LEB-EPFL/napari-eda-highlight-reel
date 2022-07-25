"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QScrollBar, QListView

from pathlib import Path
import scipy.ndimage as ndi

if TYPE_CHECKING:
    import napari

class Extractor_Widget(QWidget):
    def __init__(self, napari_viewer, nbh_size = 10):
        super().__init__()
        self.viewer = napari_viewer
        
        self.setLayout(QVBoxLayout)
        self.nbh_size = nbh_size
        self.time_data = None

        self.events = []
        self.croppers = []
        self.eda_layer = None

        self.create_threshold_scroller()

        self.create_top_buttons()

        self.event_list = QListView()

        self.create_bottom_buttons()
        
        self.addWidget(self.thresh_scroller)
        self.layout.addLayout(self.top_btn_layout)
        self.layout.addWidget(self.event_list)
        self.layout.addLayout(self.bottom_btn_layout)

    def create_threshold_scroller(self):
        self.thresh_scroller= QScrollBar(Qt.Horizontal)
        self.thresh_scroller.setMinimum(0)
        self.thresh_scroller.setSingleStep(1)
        self.thresh_scroller.setMinimumWidth(150)
    
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
        

    def scan(self):
        open_events = []
        framenumber = len(self.eda_layer)
        for i in framenumber:
            actualist = find_cool_thing_in_frame(self.eda_layer.data[i,0], self.nbh_size)
            while actualist['x'] and actualist['y']:
                new_event = True
                for ev in open_events:
                    if abs(ev.c_x - actualist['x'][0])<self.nbh_size and abs(ev.c_y - actualist['y'][0])<self.nbh_size and ev.last_frame == i-1:
                        ev.last_frame = i
                        new_event = False
                if new_event:
                    open_events.append(Event([actualist['x'],actualist['y']],i))
                actualist['x'].pop(0)
                actualist['x'].pop(0)
        return open_events
                    
                    



def find_cool_thing_in_frame(frame,threshold: float, nbh_size):
    data_max = ndi.filters.maximum_filter(frame, nbh_size)
    maxima = (frame == data_max)
    labeled, num_objects = ndi.label(maxima)
    slices = ndi.find_objects(labeled)
    Events_centers = {'x': [], 'y': []}
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        Events_centers['x'].append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        Events_centers['y'].append(y_center)
    return Events_centers



class Event():
    def __init__(self,center_position, first_frame):
        self.c_x = center_position[0]
        self.c_y = center_position[1]
        self.first_frame = first_frame
        self.last_frame = first_frame

    
    

class Cropper_Widget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer, center_position, first_frame, last_frame):
        super().__init__()
        self.viewer = napari_viewer
        self.center_position = center_position
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.time_data = None
        self.event_scores = None

        self.size_slider = QSlider()
        self.hister_slider = QSlider()

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

    def crop(self,layer_numbers,sizes,hf_frame, hl_frame):
        to_use = self.convert_to_easy_format(layer_numbers)
        return centered_crop(to_use, self.center_position,sizes,hf_frame,hl_frame)

    def convert_to_easy_format(self, layer_numbers):
        ready_video = dict()
        for i in layer_numbers:
            video = np.array(self._viewer.layers[i].data)
            video = video.squeeze()
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
def centered_crop(videos: dict, Center_pos, Sizes, frame_begin, frame_end): #bref, center pos et Sizes seront donnes dans l'ordre y x
        limits = []
        limits.append([frame_begin,frame_end])
        for i in range(len(Center_pos)):
            frst = int(Center_pos[i] - 0.5*Sizes[i])
            lst = int(Center_pos[i] + 0.5* Sizes[i])
        #make the limits
        new_lay = dict()
        for key in videos.keys:
            new_lay[key] = layer_crop(videos[key],limits)
        return 0

def layer_crop(image: np.array, limits: list):
        if image.ndim() == len(limits):
            newimg = image[limits[0][0]:limits[0][1],limits[1][0]:limits[1][1],limits[2][0]:limits[2][1]]
        else:
            raise
        return newimg

def connect_eda(widget):
   edapath = str(Path(widget.image_path).parent / 'EDA')
   widget._viewer.open(edapath, plugin = "napari-ome-zarr")
   widget._viewer.layers[-1].opacity = 0.25
   widget._viewer.layers[-1].data = widget._viewer.layers[-1].data[:-1]