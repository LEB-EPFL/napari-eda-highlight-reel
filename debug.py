import napari
import os

viewer = napari.Viewer()
path=  str(os.path.dirname(__file__))+'/images/steven_192.ome.zarr/Images'#"https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr/"
viewer.open(path, plugin = 'napari-ome-zarr')
napari.run()