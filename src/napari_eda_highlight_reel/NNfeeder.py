"""Module for outsourcing the preprocessing of raw data to prepare for neural network.
Rescale to 81 nm/px, background subtraction and contrast enhancement.
Returns:
    [type]: [description]
"""


from typing import Any
import numpy as np
from skimage import exposure, filters, transform

from .ImageTiles import getTilePositionsV2


def prepareNNImages(imgs: dict, specs: dict, model, bacteria=False):
    """Preprocess raw iSIM images before running them throught the neural network.
    Args:
        bact_img ([type]): full frame of the mito data as numpy array
        ftsz_img ([type]): full frame of the drp data as numpy array
        nnImageSize ([type]): image size that is needed for the neural network. Default is 128
    Returns:
        [type]: Returns a 3D numpy array that contains the data for the neural network and the
        positions dict generated by getTilePositions for tiling.
    """
    # Set iSIM specific values
    pixelCalib = 56  # nm per pixel
    sig = 121.5/81  # in pixel
    resizeParam = pixelCalib/81  # no unit
    try:
        nnImageSize = model.layers[0].input_shape[0][1]
    except AttributeError:
        nnImageSize = model
    positions = None

    # Preprocess the images
    if nnImageSize is None or imgs[1].shape[1] > nnImageSize:
        # Adjust to 81nm/px
        for id in imgs.keys():
            if specs[id]['ISIM Rescale']:
                imgs[id] = transform.rescale(imgs[id], resizeParam)
            if specs[id]['Gaussian Filter']:
                imgs[id] = filters.gaussian(imgs[id], sig, preserve_range=True)
            # This leaves an image that is smaller then initially

            # gaussian and background subtraction
            if specs[id]['Remove Background']:
                imgs[id] = imgs[id] - filters.gaussian(imgs[id], sig*5, preserve_range=True)
        

        # Tiling
        if nnImageSize is not None:
            positions = getTilePositionsV2(imgs[1], nnImageSize)
            contrastMax = 255
        else:
            contrastMax = 1

        # Contrast
        for id in imgs.keys():
            if specs[id]['Normalize 0-1'] or specs[id]['Normalize 0-255']:
                imgs[id] = exposure.rescale_intensity(imgs[id], (np.amin(imgs[id]), np.amax(imgs[id])), out_range=(0, contrastMax))

    else:
        positions = {'px': [(0, 0, imgs[1].shape[1], imgs[1].shape[1])],
                     'n': 1, 'overlap': 0, 'stitch': 0}

    # Put into format for the network
    if nnImageSize is not None:
        for id in imgs.keys():
            imgs[id] = imgs[id].reshape(1, imgs[id].shape[0], imgs[id].shape[0], 1)
        inputDataFull = np.concatenate(imgs.values(), axis=3)

        # Cycle through these tiles and make one array for everything
        i = 0
        inputData = np.zeros((positions['n']**2, nnImageSize, nnImageSize, 2), dtype=np.uint8())
        for position in positions['px']:

            inputData[i, :, :, :] = inputDataFull[:,
                                                  position[0]:position[2],
                                                  position[1]:position[3],
                                                  :]
            if bacteria:
                inputData[i, :, :, 1] = exposure.rescale_intensity(
                    inputData[i, :, :, 1], (0, np.max(inputData[i, :, :, 1])),
                    out_range=(0, 255))

            inputData[i, :, :, 0] = exposure.rescale_intensity(
                inputData[i, :, :, 0], (0, np.max(inputData[i, :, :, 0])),
                out_range=(0, 255))
            i = i+1
        inputData = inputData.astype('uint8')
    else:
        # This is now missing the tile-wise rescale_intensity for the mito channel.
        # Image shape has to be in multiples of 4, not even quadratic
        for id in imgs.keys():
            cropPixels = (imgs[id].shape[0] - imgs[id].shape[0] % 4, imgs[id].shape[1] - imgs[id].shape[1] % 4)
            imgs[id] = imgs[id][0:cropPixels[0], 0:cropPixels[1]]

        positions = getTilePositionsV2(imgs[1], 128)
        for id in imgs.keys():
            imgs[id] = imgs[id].reshape(1, imgs[id].shape[0], imgs[id].shape[0], 1)
        inputData = np.stack(list(imgs.values()), 3)

    return inputData, positions