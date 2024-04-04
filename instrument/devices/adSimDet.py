"""
Simulated detector
"""

from ophyd.areadetector import SimDetector, SimDetectorCam, SingleTrigger, MultiTrigger
from ophyd.areadetector.plugins import (
    StatsPlugin,
    ImagePlugin,
    ROIPlugin,
    HDF5Plugin,
    ProcessPlugin,
)
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector.plugins import ROIPlugin_V34, StatsPlugin_V34
from ophyd import Component as Cpt
from ophyd import Kind
from ophyd import EpicsMotor

from ..session_logs import logger

logger.info(__file__)

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    pass

class LocalSimDetector(SimDetector):    
    _default_configuration_attrs = ('roi1', 'roi2', 'roi3', 'roi4')
    _default_read_attrs = ('cam', 'stats1', 'stats2', 'stats3', 'stats4')
    
    image = Cpt(ImagePlugin, 'simDet:image1:')
    cam = Cpt(SimDetectorCam, 'simDet:cam1:')
    transform_type = 0
#   hdf5 = Cpt(HDF5PluginWithFileStore, 'HDF1:',
#            write_path_template='/home/beams/MWYMAN/data',
#            read_path_template='/home/beams/MWYMAN/images',
#            read_attrs=[],
#            root='/')
    stats1 = Cpt(StatsPlugin_V34, 'Stats1:')
    stats2 = Cpt(StatsPlugin_V34, 'Stats2:')
    stats3 = Cpt(StatsPlugin_V34, 'Stats3:')
    stats4 = Cpt(StatsPlugin_V34, 'Stats4:')
#   stats5 = Cpt(StatsPlugin, 'Stats5:')
    roi1 = Cpt(ROIPlugin_V34, 'ROI1:')
    roi2 = Cpt(ROIPlugin_V34, 'ROI2:')
    roi3 = Cpt(ROIPlugin_V34, 'ROI3:')
    roi4 = Cpt(ROIPlugin_V34, 'ROI4:')
#   proc1 = Cpt(ProcessPlugin, 'Proc1:')
    
    
# TODO -- maybe this is where statsplugin stuff is added?   
    def default_kinds(self):
        _remove_from_config = (
            "file_number_sync",  # Removed from EPICS
            "file_number_write",  # Removed from EPICS
            "pool_max_buffers",  # Removed from EPICS
            # Following were removed for Eiger X 500
            "fw_clear", 
            "link_0",
            "link_1",
            "link_2",
            "link_3",
            "dcu_buff_free",
            # all below are numpy.ndarray
            "configuration_names",
            "stream_hdr_appendix",
            "stream_img_appendix",
            "dim0_sa",
            "dim1_sa",
            "dim2_sa",
            "nd_attributes_macros",
            "dimensions",
            'asyn_pipeline_config',
            'dim0_sa',
            'dim1_sa',
            'dim2_sa',
            'dimensions',
            'histogram',
            'ts_max_value',
            'ts_mean_value',
            'ts_min_value',
            'ts_net',
            'ts_sigma',
            'ts_sigma_xy',
            'ts_sigma_y',
            'ts_total',
            'ts_timestamp',
            'ts_centroid_total',
            'ts_eccentricity',
            'ts_orientation',
            'histogram_x',
        )
        self.cam.configuration_attrs += [
            item for item in SimDetectorCam.component_names if item not in
            _remove_from_config
            ]

        for name in self.component_names:
            comp = getattr(self, name)
            if isinstance(comp, (ROIPlugin_V34, StatsPlugin_V34)):
                comp.configuration_attrs += [
                    item for item in comp.component_names if item not in
                    _remove_from_config
                    ]
            if isinstance(comp, StatsPlugin_V34):
                comp.total.kind = Kind.hinted #<-- makes total count a BEC plot
                comp.read_attrs += ["max_value", "min_value"]
# TODO add case for Process plugin or is this unnecessary?
#       if isinstance(comp, ProcPlugin_V34): 
# Place holder if needed later  
    def default_settings(self):
        pass
    
class LocalSimDetSingle(SingleTrigger, LocalSimDetector):
    pass

class LocalSimDetMulti(MultiTrigger, LocalSimDetector):
	pass




