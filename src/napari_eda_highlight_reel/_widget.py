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

import tifffile
import xmltodict

import napari

class Extractor_Widget(QWidget):
    def __init__(self, napari_viewer, nbh_size = 10):
        super().__init__()
        self._viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.nbh_size = nbh_size
        self.time_data = None
        self.image_path = None

        self.eda_layer = None
        self.eda_ready = False
        self.max_ev_score = None
        self.threshold = 0

        self.create_threshold_scroller()

        self.create_top_buttons()

        self.event_list = QListWidget()

        self.create_bottom_buttons()
        
        self.layout().addLayout(self.thresh_grid)
        self.layout().addLayout(self.top_btn_layout)
        self.layout().addWidget(self.event_list)
        self.layout().addLayout(self.bottom_btn_layout)

        self.init_data()

        self.scan_btn.clicked.connect(self.full_scan)
        self.add_btn.clicked.connect(self.create_new_event)

        

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
        connect_xml_metadata(self._viewer)
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
        dial = EDA_name_Dialog()
        dial.exec()

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



class EDA_Event():
    def __init__(self,name, center_position, first_frame):
        self.name = name
        self.c_x = center_position[0]
        self.c_y = center_position[1]
        self.c_z = center_position[2]
        self.first_frame = first_frame-1
        self.last_frame = first_frame

class EDA_name_Dialog(QDialog):
    def __init__(self,widget: Extractor_Widget):
        super().__init__()
        self._widget = widget
        self.setLayout(QGridLayout)
        self.label = QLabel("Enter EDA Image layer's name")
        self.inser = QLineEdit()
        self.canc_button = QPushButton('Cancel')
        self.sel_button = QPushButton('Select name')
        self.layout().addWidget(self.label,0,0)
        self.layout().addWidget(self.inser,0,1)
        self.layout().addWidget(self.canc_button,1,0)
        self.layout().addWidget(self.sel_button,1,1)

        self.canc_button.clicked.connect(self.cancel_name_selection)
        self.sel_button.clicked.connect(self.select_name)

    def cancel_name_selection(self):
        self._widget.eda_ready = False
        self._widget.eda_layer = 'None'
        self.reject()

    def select_name(self):
        chosen_name = self.inser.text()
        for lay in self._widget._viewer.layers:
            if lay.name == chosen_name:
                self._widget.eda_ready = True
                self._widget.eda_layer = lay
                self.accept()
        if self._widget.eda_ready ==False:
            errmess = QErrorMessage()
            errmess.showMessage('Please select the name of one of the available layers')



class Cropper_Widget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, extractor: Extractor_Widget, event: EDA_Event):
        super().__init__()
        self._extractor: Extractor_Widget = extractor
        self._event = event
        self.time_data = None
        self.event_scores = None
        self.layers_to_crop_indexes = self.get_image_layers_indexes()

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
        i = 0
        for lay in self._extractor._viewer.layers:
            self.layermenu.addAction(lay.name)
            self.layermenu.actions()[-1].setCheckable(True)
            if i in self.layers_to_crop_indexes:
                self.layermenu.actions()[-1].setChecked()
            i = i+1

    def update_croplist_from_layermenu(self):
        self.layers_to_crop_indexes = []
        for i in range(len(self.layermenu.actions())):
            if self.layermenu.actions()[i].isChecked():
                self.layers_to_crop_indexes.append(i)

    def create_top_lane(self):
        self.top_lane = QHBoxLayout()
        self.nameLabel = QLineEdit(self._event.name)

        self.layerbutton = QToolButton(self)
        self.layerbutton.setText('Layers')
        self.layermenu = QMenu(self)
        self.layerbutton.setMenu(self.layermenu)
        self.layerbutton.setPopupMode(QToolButton.InstantPopup)
        self.update_layermenu()

        for i in range(len(self._extractor._viewer.layers)):
            if i in self.layers_to_crop_indexes:
                    self.layermenu.actions()[i].setChecked(True)
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

    def get_layers_data(self, layer_numbers):
        benedict = {}
        for i in layer_numbers:
            datas = layer_modes(self._extractor._viewer.layers[i])
            benedict[datas['name']] = datas
        return benedict

    def get_image_layers_indexes(self):
        lili = []
        for i in range(len(self._extractor._viewer.layers)):
            if type(self._extractor._viewer.layers[i]) == napari.layers.image.image.Image:
                lili.append(i)
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

        



    def full_crop(self,layer_numbers, sizes,hf_frame, hl_frame):
        to_use = self.convert_to_easy_format(layer_numbers)
        return centered_crop(to_use, self.center_position,sizes,hf_frame,hl_frame)

    def convert_to_easy_format(self, layer_numbers):
        """
        takes the layers images and convert them to a dictionary of 4-dimensional numpy arrays
        """
        ready_video = dict()
        for i in layer_numbers:
            video = np.asarray(self._extractor._viewer.layers[i].data)
            if video.ndim == 3:
                video = np.expand_dims(video, axis = 1)
            ready_video[self._extractor._viewer.layers[i].name] = video
        return ready_video

    def save_reel(self):
        return 0

    def view_reel(self):
        all_layers = range(len(self._extractor._viewer.layers))
        new_lay = {}
        vids = self.convert_to_easy_format(all_layers)
        for key in vids.keys():
            new_lay[key] = layer_crop(vids[key],self.get_corrected_limits())
        new_view = napari.Viewer()
        dats = self.get_layers_data(all_layers)
        for i in range(len(new_lay.keys())):
            new_view.add_image(new_lay[list(new_lay.keys())[i]])
            mods_to_layer(new_view.layers[i],dats[list(new_lay.keys())[i]])
            if len(self._extractor._viewer.layers[i].metadata.keys()) > 0:
                new_view.layers[i].metadata = self.crop_ome_metadata(self._extractor._viewer.layers[i].metadata)

    def pass_metadata(self):
        """
        Modify the met['OME']['Image']['Pixels']['TiffData'] and met['OME']['Image']['Pixels']['Planes'] taking only the frames
        in the reel, modify sizes names and positions"""
        new_meta = {}
        for lay in self._extractor._viewer.layers[self.layers_to_crop_indexes]:
            new_meta[lay.name] = self.crop_ome_metadata(lay.metadata)
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

    


def centered_crop(videos: dict, Center_pos, Sizes, frame_begin, frame_end): #bref, center pos et Sizes seront donnes dans l'ordre z y x
    """
    Crop the layer with given center point and desired size of the cropped region
    """
    limits = []
    limits.append([frame_begin,frame_end])
    for i in range(len(Center_pos)):
        frst = Center_pos[i] - int(0.5*Sizes[i])
        lst = Center_pos[i] + int(0.5*Sizes[i])
        limits.append([frst,lst])
    new_lay = dict()
    for key in videos.keys():
        new_lay[key] = layer_crop(videos[key],limits)
    return new_lay

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

def connect_eda(widget):
    """
    Connect the associated EDA images in a new layer
    """
    edapath = str(Path(widget.image_path).parent / 'EDA')
    widget._viewer.open(edapath, plugin = "napari-ome-zarr")
    widget._viewer.layers[-1].blending = 'additive'
    widget._viewer.layers[-1].name = 'EDA'

def get_dict_from_ome_metadata(usepath: Path):
   if usepath.suffix == '.tif':
         with tifffile.TiffFile(usepath) as tif:
            XML_metadata= tif.ome_metadata #returns a reference to a function that accesses the metadata as a OME XML file
   else:
      metapath = str(usepath.parent / 'OME' / 'METADATA.ome.xml')
      XML_metadata = open(metapath,'r').read()
   dict_metadata=xmltodict.parse(XML_metadata) #converts the xml to a dictionary to be readable
   return dict_metadata

def get_times_from_dict(dict_metadata: dict, channel = 0):
   times=[]
   num_pages = len(dict_metadata['OME']['Image']['Pixels']['Plane'])
   for frame in range(0,num_pages):
      #time should be in either s or ms
      if float(dict_metadata['OME']['Image']['Pixels']['Plane'][frame]['@TheC'])==channel: #checks if correct channel
         frame_time_unit=dict_metadata['OME']['Image']['Pixels']['Plane'][frame]['@DeltaTUnit']
         if frame_time_unit== 's' :
            convert_unit_to_ms=1000
            times.append(convert_unit_to_ms*float(dict_metadata['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))
         elif frame_time_unit == 'ms':
            convert_unit_to_ms=1
            times.append(convert_unit_to_ms*float(dict_metadata['OME']['Image']['Pixels']['Plane'][frame]['@DeltaT']))
         else:
            print('Time units not in ms or s but in '+ frame_time_unit+'. A conversion to ms or s must be done.')
   times = [x - times[0] for x in times] #remove any offset from time
   return times

def get_times(widget):
   """Method that gets the capture time from the metadata.
   
   Input:-
   Output: Vector of time metadata [ms] of images found at given image path and channel.

   The times of each image stack from a ome.tif file is read in [ms] or [s] and then returned in [ms]. 
   The times are taken from a given channel. The data can only be read from an ome.tif file. The Offset 
   from time t=0 subrtracted to the times before it is returned.
   The following code is inspired from the solution for reading tiff files metadata from Willi Stepp.
   """

   dict_metadata=get_dict_from_ome_metadata(Path(widget.image_path)) #converts the xml to a dictionary to be readable
   times = get_times_from_dict(dict_metadata, channel=widget.channel) #remove any offset from time
   return times

def layer_modes(layer) -> dict:
    """
    save the proerties of a layer in a dictionary
        self.setMinimumHeight(200)
    """
    datas = {}
    datas['name'] = layer.name
    datas['opacity'] = layer.opacity
    datas['contrast_limits'] = layer.contrast_limits
    datas['gamma'] = layer.gamma
    datas['colormap'] = layer.colormap
    datas['blending'] = layer.blending
    datas['interpolation'] = layer.interpolation
    return datas

def connect_xml_metadata(viewer: napari.Viewer):
    for lay in viewer.layers:
        impath = Path(lay.source.path)
        if impath.suffix != 'tif':
            metapath = impath.parent / 'OME' / 'METADATA.ome.xml'
            XML_metadata = open(metapath,'r').read()
            lay.metadata = xmltodict.parse(XML_metadata)
            


def mods_to_layer(layer: napari.layers.image.image.Image, mods: dict):
    """
    gives the properties defined in mods to the layer
    """
    layer.name = mods['name']
    layer.opacity = mods['opacity']
    layer.contrast_limits = mods['contrast_limits']
    layer.gamma = mods['gamma']
    layer.colormap = mods['colormap']
    layer.blending = mods['blending']
    layer.interpolation = mods['interpolation']

