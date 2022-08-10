import napari
import os

viewer = napari.Viewer()
path=  str(os.path.dirname(__file__))+'/images/steven_192.ome.zarr/Images'#"https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr/"
#path= str(os.path.dirname(__file__))+'/images/example_image.tif'
viewer.open(path, plugin = 'napari-ome-zarr')
viewer.window.add_plugin_dock_widget('napari-eda-highlight-reel','Extractor Widget') #'Add time scroller'
napari.run()