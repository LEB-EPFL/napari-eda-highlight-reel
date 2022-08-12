import numpy as np

import napari_eda_highlight_reel._widget as reel




# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams


############ Test Cropper Auxiliary ################

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



