import numpy as np

import napari_eda_highlight_reel._widget as reel




# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams

def test_layer_crop():
    dims = [np.random.randint(0,100),np.random.randint(0,100),np.random.randint(0,100),np.random.randint(0,100)]
    image = np.random.rand(dims[0],dims[1],dims[2],dims[3])
    limits = []
    sizes = []
    timl = sorted([np.random.randint(0,dims[0]),np.random.randint(0,dims[0])])
    limits.append(sorted([np.random.randint(0,dims[1]),np.random.randint(0,dims[1])]))
    limits.append(sorted([np.random.randint(0,dims[2]),np.random.randint(0,dims[2])]))
    limits.append(sorted([np.random.randint(0,dims[3]),np.random.randint(0,dims[3])]))
    sizes.append(timl[1]-timl[0])
    for i in range(len(limits)):
        sizes.append(limits[i][1]-limits[i][0])
    newimg = reel.layer_crop(image,timl,limits)
    assert newimg.shape == tuple(sizes)

def test_centered_crop():
    videos = dict()
    videos['First'] = np.random.rand(50,50,50,50)
    videos['Second'] = np.random.rand(50,50,50,50)
    videos['Third'] = np.random.rand(50,50,50,50)
    center_pos = [13,5,38]
    barr = []
    sizes = []
    for i in range(len(center_pos)):
        barr.append(2*min([center_pos[i],50-center_pos[i]]) if videos['First'].shape[i+1] > 1 else 1)
        sizes.append(np.random.randint(0,barr[i]))
    tmp_l = sorted([np.random.randint(0,50),np.random.randint(0,50)])
    cropped = reel.centered_crop(videos, center_pos, sizes, tmp_l[0], tmp_l[1])
    for key in videos.keys():
        assert cropped[key].shape == tuple([(tmp_l[1]-tmp_l[0]),sizes[0],sizes[1],sizes[2]])

def test_centered_crop_2d():
    videos = dict()
    videos['First'] = np.random.rand(50,1,50,50)
    videos['Second'] = np.random.rand(50,1,50,50)
    videos['Third'] = np.random.rand(50,1,50,50)
    center_pos = [0,5,38]
    barr = []
    sizes = []
    for i in range(len(center_pos)):
        barr.append(2*min([center_pos[i],videos['First'].shape[i+1]-center_pos[i]]) if videos['First'].shape[i+1] > 1 else 1)
        sizes.append(np.random.randint(0,barr[i]))
    tmp_l = sorted([np.random.randint(0,50),np.random.randint(0,50)])
    cropped = reel.centered_crop(videos, center_pos, sizes, tmp_l[0], tmp_l[1])
    for key in videos.keys():
        assert cropped[key].shape == tuple([(tmp_l[1]-tmp_l[0]),sizes[0],sizes[1],sizes[2]])
