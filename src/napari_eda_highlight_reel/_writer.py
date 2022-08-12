"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import os
import shutil

import napari
import numpy as np
from ome_zarr import io
from ome_zarr import writer
import zarr

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union
from pathlib import Path

import dict2xml

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_image(path: str, data: Any, meta: dict):
    """Writes a single image layer"""
    write_ngff_image(path,np.asarray(data))
    return path


def write_multiple(path: str, data: List[FullLayerData]):
    """Writes multiple layers of different types."""
    eda_layer = None
    omedata = []
    for i in range(len(data)):
        if data[i][1]['metadata'].__contains__('EDA') and data[i][1]['EDA']:
            eda_layer = data[i]
        if data[i][1]['metadata'].__contains__('OME'):
            omedata.append(data[i][1]['metadata'])
    eda_path = path + '/EDA'
    img_path = path + '/Images'
    ome_path = path + '/OME/METADATA.ome.xml'
    if eda_layer is not None:
        write_ngff_image(eda_path, eda_layer[0], axes = "tzyx")
    if len(omedata) > 0:
        to_write = merge_layers_ngff(omedata)
        write_ngff_image(img_path, to_write)
        xmlstr = dict2xml.dict2xml(omedata)
        flot = open(ome_path,'w')
        flot.write(xmlstr)
        flot.close
    return [eda_path, img_path, ome_path]
    #write_ngff_image(path, merge_layers_ngff(data))

#def write_full_eda_format(path: str, data: List[FullLayerData]):
def write_multiple_again(path: str, data: List[FullLayerData]):
    if os.path.isdir(path):
        shutil.rmtree(path)
    omedata = []
    eda_layer = None
    os.mkdir(path)
    for i in range(len(data)):
        if data[i][1]['metadata'].__contains__('EDA'):
            if data[i][1]['metadata']['EDA']:
                eda_layer = data[i]
        if data[i][1]['metadata'].__contains__('OME'):
            omedata.append(data[i])
    eda_path = path + '/EDA'
    img_path = path + '/Images'
    omedir = path + '/OME'
    os.mkdir(omedir)
    ome_path = omedir + '/METADATA.ome.xml'
    if eda_layer is not None:
        write_ngff_image(eda_path, np.expand_dims(np.asarray(eda_layer[0]),axis = 1))
    if len(omedata) > 0:
        to_write = merge_layers_ngff(omedata)
        write_ngff_image(img_path, to_write)
        xmlstr = dict2xml.dict2xml(omedata[0][1]['metadata'])
        flot = open(ome_path,'w')
        flot.write(xmlstr)
        flot.close
    return [eda_path, img_path, ome_path]

    



def merge_layers_ngff(dats: List[FullLayerData]) -> np.ndarray:
    """
    Merge the layers in a multi channel image.
    Being the image in the ngff format the order of the coordinates will be t c z y x
    """
    shp = dats[0][0].shape
    final = np.ndarray([shp[0],len(dats),shp[1],shp[2],shp[3]])
    if check_uniform_dimensions(dats):
        for i in range(len(dats[0][0])):
            for j in range(len(dats)):
                final[i,j] = np.asarray(dats[j][0][i])
    else:
        print('layers not uniform')
    return final

def check_uniform_dimensions(dats: List[FullLayerData]) -> bool:
    if len(dats) > 1:
        shp = dats[0][0].shape
        for dada in dats:
            if dada[0].shape != shp:
                return False
    return True


def write_ngff_image(path: str, image: np.ndarray, axes="tczyx"):
    # write the image data
    store = io.parse_url(path, mode="w").store
    root = zarr.group(store=store)
    writer.write_image(image=image, group=root, axes=axes, scaler = None)
    


def _zarr_group(path: str, name: str = None) -> zarr.Group:
        path = path + "/" + name if name is not None else path
        store = io.parse_url(path, mode="w").store
        root = zarr.group(store=store)
        return root