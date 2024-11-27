import napari
import os

viewer = napari.Viewer()
#path=  str(os.path.dirname(__file__))+'/images/steven_14_MMStack_Injection.ome.tif'                 #/steven_192.ome.zarr/Images'#"https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/9836842.zarr/"
#savepath = str(os.path.dirname(__file__))+'/images/steven_192.ome.zarr/Reels/a.eda'

# path = "W:/Scientific_projects/deep_events_WS/data/phaseEDA/20240417_phaseEDA/FOV_005.ome.zarr",
# viewer.open(path, plugin = 'napari-ome-zarr')
# _, extractor = viewer.window.add_plugin_dock_widget('napari-eda-highlight-reel','Extractor Widget')
# extractor.thresh_scroller.setValue(8800)
# extractor.full_scan()

#viewer.layers.save(savepath, plugin='napari-eda-highlight-reel')
napari.run()