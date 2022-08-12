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
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QWidget, QScrollBar, QListWidget, QListWidgetItem, QDialog, QLineEdit,QErrorMessage, QComboBox, QMenu, QToolButton
from qtpy.QtCore import Qt, QTimer
from qtpy import QtGui
import dask.array

from pathlib import Path
import scipy.ndimage as ndi

from ._writer import write_multiple_again

import tifffile
import xmltodict
import dicttoxml

import napari
from napari.types import FullLayerData

##################### Extractor Widget ####################

class Extractor_Widget(QWidget):
    """This doct widget is an extractor in which, given the layer having the EDA information and the event score threshold, creates the events

    Parameters
    ----------

    napari_viewer : napari.Viewer
        the viewer that the extractor will extract the important events from
    """
    def __init__(self, napari_viewer, nbh_size = 10):
        super().__init__()
        self._viewer: napari.Viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.nbh_size = nbh_size
        self.time_data = None
        self.image_path = None

        self.eda_layer = None
        self.eda_ready = False
        self.max_ev_score = None
        self.threshold = 0

        self.create_EDA_layer_selector()

        self.create_threshold_scroller()

        self.create_top_buttons()

        self.event_list = QListWidget()

        self.create_bottom_buttons()
        
        self.layout().addLayout(self.choose_eda_line)
        self.layout().addLayout(self.thresh_grid)
        self.layout().addLayout(self.top_btn_layout)
        self.layout().addWidget(self.event_list)
        self.layout().addLayout(self.bottom_btn_layout)

        self.init_data()

        self.scan_btn.clicked.connect(self.full_scan)
        self.add_btn.clicked.connect(self.create_new_event)

    def create_EDA_layer_selector(self):
        """Creates the selector for the EDA layer"""
        self.choose_eda_line = QHBoxLayout()
        self.choose_eda_line.addWidget(QLabel('EDA image layer'))
        self.eda_layer_chooser = QComboBox()
        for lay in self._viewer.layers:
            self.eda_layer_chooser.addItem(lay.name)
        self.choose_eda_line.addWidget(self.eda_layer_chooser)
        

    def update_eda_layer_from_chooser(self, text):
        self.eda_layer = self._viewer.layers[text]
        self.set_max_thresh()

    def update_eda_layer_chooser(self):
        self.eda_layer_chooser.clear()
        for lay in self._viewer.layers:
            self.eda_layer_chooser.addItem(lay.name)

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
        self.save_all_btn.clicked.connect(self.save_all_events)
        self.view_all_btn.clicked.connect(self.view_all_events)

    def init_data(self):
        """Initialize data from the layers"""
        if self.image_path != self._viewer.layers[0].source.path : #update data if new source is added
            self.image_path = self._viewer.layers[0].source.path
        #self.time_data=get_times(self)#init times of initial image
        connect_xml_metadata(self._viewer)
        connect_eda(self)
        self.update_eda_layer_chooser()
        self.search_eda_layer()
        self.eda_layer_chooser.currentTextChanged.connect(self.update_eda_layer_from_chooser)
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
                try:
                    self.eda_layer_chooser.setCurrentText('EDA')
                except:
                    print('No layer named EDA in the selector')
        if not self.eda_ready:
            self.ask_eda_layer_name()
            

    def create_new_event(self):
        evvy = EDA_Event('Event ' + str(self.event_list.count()), [0,0,0],1)
        new_crp = Cropper_Widget(self,evvy)
        item = QListWidgetItem()
        item.setSizeHint(new_crp.sizeHint())
        self.event_list.addItem(item)
        self.event_list.setItemWidget(item,new_crp)


    def full_scan(self):
        self.event_list.clear()
        if self.eda_ready:
            events = self.basic_scan(self.eda_layer)
            for ev in events:
                item = QListWidgetItem()
                cropper = Cropper_Widget(self, ev)
                item.setSizeHint(cropper.sizeHint())
                self.event_list.addItem(item)
                self.event_list.setItemWidget(item,cropper)
    

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
                    open_events.append(EDA_Event('Event ' + str(len(open_events)),[actualist[0]['x'],actualist[0]['y'],actualist[0]['z']],i))
                actualist.pop(0)
            print('frame number ' + str(i) + 'scanned')
        for ev in open_events:
            ev.last_frame = ev.last_frame+1
        return open_events
    
    def update_threshhold(self):
        self.threshold = self.thresh_scroller.value()*self.max_ev_score/100
        self.thresh_show.setText(str(self.threshold))
    
    def save_all_events(self):
        for i in range(self.event_list.count()):
            self.event_list.itemWidget(self.event_list.item(i)).save_reel()

    def view_all_events(self):
        for i in range(self.event_list.count()):
            self.event_list.itemWidget(self.event_list.item(i)).view_reel()

################### Auxiliary functions for the Extractor Widget ################

def find_cool_thing_in_frame(frame, threshold: float, nbh_size: int) -> list[dict[str:float]]:
    """Function that takes a 3D frame and takes a list of the positions of the local maxima that are higher than the threshold
    
    Parameters
    ----------
    
    frame : numpy.ndarray
        The 3D image to be analyzed
    threshold : float
        Minimal value of the event score a maxima must have to be considered
    nbh_size : int
        Size of the neighbourhood around a local mximum wich it is considered that an other local maximum would be due to noise

    Returns
    -------

    list of dictionaries having at the entries 'x', 'y' and 'z' the x, y nd z coordinate of every event center
    """
    data_max = ndi.maximum_filter(frame, nbh_size, mode = 'constant', cval = 0)
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

 ###################### EDA Event Structure ########################

class EDA_Event():
    """ Sctucture that represents an interesting event in a 3D video"""
    def __init__(self,name, center_position, first_frame):
        self.name = name
        self.c_x = center_position[0]
        self.c_y = center_position[1]
        self.c_z = center_position[2]
        self.first_frame = first_frame-1
        self.last_frame = first_frame


######################## Cropper Widget #########################


class Cropper_Widget(QWidget):
    """Widget that make it possible to crop and then view separately or save an event
    
    Parameters
    ----------

    extractor : Ectractor_Widget
        the extractor that created the cropper
    event : EDA_Event
        the event at which the cropper is associated
    """
    def __init__(self, extractor: Extractor_Widget, event: EDA_Event):
        super().__init__()
        self._extractor: Extractor_Widget = extractor
        self._event = event
        self.time_data = None
        self.event_scores = None
        self.layers_to_crop_names = self.get_image_layers_names()

        self.max_crop_sizes = {'x': self._extractor.eda_layer.data.shape[3], 'y': self._extractor.eda_layer.data.shape[2], 'z': self._extractor.eda_layer.data.shape[1]}
        self.crop_sizes = {'x': min(100,self.max_crop_sizes['x']), 'y': min(100,self.max_crop_sizes['y']), 'z': min(100,self.max_crop_sizes['z'])}

        self.create_top_lane()
        self.view_btn = QPushButton('View')
        self.save_btn = QPushButton('Save')
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.view_btn)
        self.btn_layout.addWidget(self.save_btn)
        self.create_val_grid()
        
                
        self.setLayout(QVBoxLayout())
        self.layout().addLayout(self.top_lane)
        self.layout().addLayout(self.val_grid)
        #self.layout().addWidget(self.size_slider)
        #self.layout().addWidget(self.hister_slider)
        self.layout().addLayout(self.btn_layout)

        self.view_btn.clicked.connect(self.view_reel)
        self.save_btn.clicked.connect(self.save_reel)

        #self.setSizePolicy()

    def __del__(self):
        indi = self.find_own_index()
        if indi >= 0:
            self._extractor.event_list.takeItem(indi)

    def find_own_index(self):
        indi = -1
        for i in range(self._extractor.event_list.count()):
            if self._extractor.event_list.itemWidget(self._extractor.event_list.item(i))._event.name == self._event.name:
                indi = i
        return indi
        

    def update_layermenu(self):
        self.layermenu.clear()
        for lay in self._extractor._viewer.layers:
            newact = self.layermenu.addAction(lay.name)
            newact.setCheckable(True)

    def update_croplist_from_layermenu(self):
        self.layers_to_crop_names = []
        for act in self.layermenu.actions():
            if act.isChecked():
                self.layers_to_crop_names.append(act.text())

    def create_top_lane(self):
        self.top_lane = QHBoxLayout()
        self.nameLabel = QLineEdit(self._event.name)

        self.layerbutton = QToolButton(self)
        self.layerbutton.setText('Layers')
        self.layermenu = QMenu(self)
        self.layerbutton.setMenu(self.layermenu)
        self.layerbutton.setPopupMode(QToolButton.InstantPopup)
        self.update_layermenu()

        for act in self.layermenu.actions():
            act.setCheckable(True)
            if act.text() in self.layers_to_crop_names:
                    act.setChecked(True)
        self.layermenu.triggered.connect(self.update_croplist_from_layermenu)

        #self._extractor._viewer.events.inserted.connect(self.update_layermenu)
        #self._extractor._viewer.events.removed.connect(self.update_layermenu)

        self.eliminate_button = QPushButton('X')
        self.eliminate_button.clicked.connect(self.__del__)
        self.nameLabel.textEdited.connect(self.update_event_name)
        self.top_lane.addWidget(QLabel('Name:'))
        self.top_lane.addWidget(self.nameLabel)
        self.top_lane.addWidget(self.layerbutton)
        self.top_lane.addWidget(self.eliminate_button)

    def update_event_name(self):
        self._event.name = self.nameLabel.text()
        self.nameLabel.setText = self._event.name

    def get_image_layers_names(self):
        lili = []
        for lay in self._extractor._viewer.layers:
            if type(lay) == napari.layers.image.image.Image:
                lili.append(lay.name)
        return lili

    def create_val_grid(self):
        self.val_grid = QGridLayout()
        self.grid_editables = {'Center position': {'x': QLineEdit(str(self._event.c_x)), 'y': QLineEdit(str(self._event.c_y)), 'z': QLineEdit(str(self._event.c_z))}, 'Size': {'x': QLineEdit(str(self.crop_sizes['x'])), 'y': QLineEdit(str(self.crop_sizes['y'])), 'z': QLineEdit(str(self.crop_sizes['z']))}, 'Frame': {'First': QLineEdit(str(self._event.first_frame)), 'Last': QLineEdit(str(self._event.last_frame))}}
        self.val_grid.addWidget(QLabel('In Space:'),0,1,1,3)
        self.val_grid.addWidget(QLabel('Center'),1,1)
        self.val_grid.addWidget(QLabel('Size'),1,2)
        self.val_grid.addWidget(QLabel('x'),2,0)
        self.val_grid.addWidget(QLabel('y'),3,0)
        self.val_grid.addWidget(QLabel('z'),4,0)
        self.val_grid.addWidget(QLabel('Frames:'),0,3)
        self.val_grid.addWidget(QLabel('First'),1,3)
        self.val_grid.addWidget(QLabel('Last'),3,3)

        self.val_grid.addWidget(self.grid_editables['Center position']['x'],2,1)
        self.val_grid.addWidget(self.grid_editables['Center position']['y'],3,1)
        self.val_grid.addWidget(self.grid_editables['Center position']['z'],4,1)

        self.val_grid.addWidget(self.grid_editables['Size']['x'],2,2)
        self.val_grid.addWidget(self.grid_editables['Size']['y'],3,2)
        self.val_grid.addWidget(self.grid_editables['Size']['z'],4,2)

        self.val_grid.addWidget(self.grid_editables['Frame']['First'],2,3)
        self.val_grid.addWidget(self.grid_editables['Frame']['Last'],4,3)

        self.grid_editables['Center position']['x'].textEdited.connect(self.update_c_p_from_grid)
        self.grid_editables['Center position']['y'].textEdited.connect(self.update_c_p_from_grid)
        self.grid_editables['Center position']['z'].textEdited.connect(self.update_c_p_from_grid)

        self.grid_editables['Size']['x'].textEdited.connect(self.update_sizes_from_grid)
        self.grid_editables['Size']['y'].textEdited.connect(self.update_sizes_from_grid)
        self.grid_editables['Size']['z'].textEdited.connect(self.update_sizes_from_grid)

        self.grid_editables['Frame']['First'].textEdited.connect(self.update_frame_limits_from_grid)
        self.grid_editables['Frame']['Last'].textEdited.connect(self.update_frame_limits_from_grid)

    
    def update_c_p_from_grid(self):
        self._event.c_x = int(self.grid_editables['Center_position']['x'].text())
        self._event.c_y = int(self.grid_editables['Center_position']['y'].text())
        self._event.c_z = int(self.grid_editables['Center_position']['z'].text())

        self.grid_editables['Center_position']['x'].setText(str(self._event.c_x))
        self.grid_editables['Center_position']['y'].setText(str(self._event.c_y))
        self.grid_editables['Center_position']['z'].setText(str(self._event.c_z))

    def update_sizes_from_grid(self):
        self.crop_sizes['x'] = int(self.grid_editables['Size']['x'].text())
        self.crop_sizes['y'] = int(self.grid_editables['Size']['y'].text())
        self.crop_sizes['z'] = int(self.grid_editables['Size']['z'].text())

        self.grid_editables['Size']['x'].setText(str(self.crop_sizes['x']))
        self.grid_editables['Size']['y'].setText(str(self.crop_sizes['y']))
        self.grid_editables['Size']['z'].setText(str(self.crop_sizes['z']))

    def update_frame_limits_from_grid(self):
        self._event.first_frame = int(self.grid_editables['Frame']['First'].text())
        self._event.last_frame = int(self.grid_editables['Frame']['Last'].text())

        self.grid_editables['Frame']['First'].setText(str(self._event.first_frame))
        self.grid_editables['Frame']['Last'].setText(str(self._event.last_frame))



    def convert_to_easy_format(self, layer_names):
        """
        takes the layers images and convert them to a dictionary of 4-dimensional numpy arrays
        """
        ready_video = dict()
        for name in layer_names:
            video = np.asarray(self._extractor._viewer.layers[name].data)
            if video.ndim == 3:
                video = np.expand_dims(video, axis = 1)
            ready_video[name] = video
        return ready_video

    def full_crop(self) -> list[FullLayerData]:
        finalist = []
        new_meta = self.pass_metadata()
        vids = self.convert_to_easy_format(self.layers_to_crop_names)
        for key in vids.keys():
            new_data = layer_crop(vids[key],self.get_corrected_limits())
            old_tuple = self._extractor._viewer.layers[key].as_layer_data_tuple()
            old_tuple[1]['metadata'] = new_meta[key]
            to_append = new_data, old_tuple[1], old_tuple[2]
            finalist.append(to_append)
        return finalist

        

    def save_reel(self):
        data = self.full_crop()
        path = str(Path(self._extractor.image_path).parent / 'Reels' / self._event.name)
        write_multiple_again(path, data)
        print(self._event.name + 'has been saved')

    def view_reel(self):
        new_lay = self.full_crop()
        new_view = napari.Viewer()
        for i in range(len(new_lay)):
            new_view.add_image(new_lay[i][0],**new_lay[i][1])

    def pass_metadata(self):
        """
        Modify the met['OME']['Image']['Pixels']['TiffData'] and met['OME']['Image']['Pixels']['Planes'] taking only the frames
        in the reel, modify sizes names and positions"""
        new_meta = {}
        for name in self.layers_to_crop_names:
            if self._extractor._viewer.layers[name].metadata.__contains__('OME'):
                new_meta[name] = self.crop_ome_metadata(self._extractor._viewer.layers[name].metadata)
            elif self._extractor._viewer.layers[name].metadata.__contains__('EDA'):
                if self._extractor._viewer.layers[name].metadata['EDA']:
                    new_meta[name] = {'EDA': True}
        return new_meta

    def crop_ome_metadata(self, ome_metadata: dict):
        cropped = ome_metadata
        limits = self.get_corrected_limits()
        sizes = []
        for i in range(len(limits)):
            sizes.append(str(limits[i][1]-limits[i][0]))
        cropped['OME']['Image']['Pixels']['@SizeT'] = sizes[0]
        cropped['OME']['Image']['Pixels']['@SizeZ'] = sizes[1]
        cropped['OME']['Image']['Pixels']['@SizeY'] = sizes[2]
        cropped['OME']['Image']['Pixels']['@SizeX'] = sizes[3]
        cropped['OME']['Image']['Pixels']['TiffData'] = [item for item in cropped['OME']['Image']['Pixels']['TiffData'] if int(item['@FirstT']) in range(limits[0][0],limits[0][1])]
        cropped['OME']['Image']['Pixels']['Plane'] = [item for item in cropped['OME']['Image']['Pixels']['Plane'] if int(item['@TheT']) in range(limits[0][0],limits[0][1])]
        return cropped

    def get_corrected_limits(self):
        limits = []
        limits.append([self._event.first_frame,self._event.last_frame+1])
        frst = self._event.c_z - int(0.5*self.crop_sizes['z'])
        lst = self._event.c_z + int(0.5*self.crop_sizes['z'])
        limits.append([frst,lst])
        frst = self._event.c_y - int(0.5*self.crop_sizes['y'])
        lst = self._event.c_y + int(0.5*self.crop_sizes['y'])
        limits.append([frst,lst])
        frst = self._event.c_x - int(0.5*self.crop_sizes['x'])
        lst = self._event.c_x + int(0.5*self.crop_sizes['x'])
        limits.append([frst,lst])
        return correct_limits(limits, np.asarray(self._extractor.eda_layer.data))

    
############################ Auxiliary functions for Cropper Widget ##########################


def layer_crop(image: np.array, limits: list):
    """
    Modify the limits in order to keep them within the image and crop the layer
    """
    limcheck = correct_limits(limits,image)
    newimg = image[limcheck[0][0]:limcheck[0][1],limcheck[1][0]:limcheck[1][1],limcheck[2][0]:limcheck[2][1],limcheck[3][0]:limcheck[3][1]]
    return newimg

def correct_limits(limits: list, image: np.array) -> list:
    """
    Modify the limits so that there are not outside the image or no dimensions ofsize 0
    """
    limcheck = limits
    for i in range(image.ndim):
        if limits[i][0] < 0:
            limcheck[i][0] = 0
        elif limits[i][0] > image.shape[i]-1:
            limcheck[i][0] = int(image.shape[i]-1)
        else:
            limcheck[i][0] = int(limits[i][0])
        if limits[i][1] > image.shape[i]:
            limcheck[i][1] = int(image.shape[i])
        elif limits[i][1] < limcheck[i][0]+1:
            limcheck[i][1] = int(limcheck[i][0]+1)
        else:
            limcheck[i][1] = int(limits[i][1])
    return limcheck

######################## Other Auxiliary Functions #####################

def connect_eda(widget):
    """
    Connect the associated EDA images in a new layer
    """
    edapath = str(Path(widget.image_path).parent / 'EDA')
    widget._viewer.open(edapath, plugin = "napari-ome-zarr")
    widget._viewer.layers[-1].blending = 'additive'
    widget._viewer.layers[-1].name = 'EDA'
    widget._viewer.layers[-1].metadata['EDA'] = True

def connect_xml_metadata(viewer: napari.Viewer):
    for lay in viewer.layers:
        impath = Path(lay.source.path)
        if impath.suffix != 'tif':
            metapath = impath.parent / 'OME' / 'METADATA.ome.xml'
            XML_metadata = open(metapath,'r').read()
            lay.metadata = xmltodict.parse(XML_metadata)