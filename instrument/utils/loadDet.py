"""
Detector loading functions
"""

__all__ = """
    loadSimDet
""".split()

from ..session_logs import logger

logger.info(__file__)

from ..devices.adSimDet import LocalSimDetSingle, LocalSimDetMulti
#from ..devices.shadowDet import ShadowDetector2D
from ..devices.shutter_simulator import shutterA, shutterB



junk_phase_template = [('junk', {shutterA:0})]                                                                      
dark_phase_template = [('dark', {shutterA:0})]                                                                      
light_phase_template = [('light', {shutterA:1})]

default_n_junk = 2      
default_n_dark = 10
default_n_light = 10
 
default_cycle = [junk_phase_template*default_n_junk + 
                 dark_phase_template*default_n_dark +
                 light_phase_template*default_n_light]

def reset_dark_light_cycle(n_junk, n_dark, n_light, num_images_sig,
                           shutter_sig = shutterA.setpoint, openVal = 1, closedVal = 0):

    junk_phase_template = [('junk', {shutter_sig:closedVal, num_images_sig:n_junk})]                                                                      
    dark_phase_template = [('dark', {shutter_sig:closedVal, num_images_sig:n_dark})]                                                                      
    light_phase_template = [('light', {shutter_sig:openVal, num_images_sig:n_light})]
    
    trig_cycle = [junk_phase_template + dark_phase_template + light_phase_template]
    
#    print('Loaded cycle: ')
#    print(trig_cycle)

    return trig_cycle


def loadSimDet(prefix="100idWYM:", multi = False, trigger_cycle = default_cycle):
    if not multi:
        print("-- Loading AD simulated Detector with single trigger")
        simDet= LocalSimDetSingle(name="simDet", prefix=prefix, read_attrs=["stats1"])
    else:
        print("-- Loading AD simulated Detector with multi-trigger")
        simDet = LocalSimDetMulti(name="simDet", prefix=prefix, 
                                  trigger_cycle = trigger_cycle, 
                                  read_attrs=["stats1"])
    
    simDet.wait_for_connection(timeout=10)
    # This is needed otherwise .get may fail!!!
    
    if multi:
        simDet.cam.stage_sigs["image_mode"] = "Multiple"
        simDet.cam.stage_sigs["num_images"] = 1
        #resetting trigger cycle to use shutter component instead of name -- this 
        #needs the detector to have been created already.  By having trigger_cycle in
        #in the keyword args, this can be skipped if the naming issue can be sorted out
        if trigger_cycle == default_cycle:
#            simDet.trigger_cycle = reset_dark_light_cycle(default_n_junk, default_n_dark,
#                                                          default_n_light, simDet.cam.shutter_control)
            simDet.trigger_cycle = reset_dark_light_cycle(default_n_junk, default_n_dark,
                                                          default_n_light, simDet.cam.num_images, 
                                                          shutter_sig = shutterA.setpoint)
    else:
        simDet.cam.stage_sigs["image_mode"] = "Single"
        simDet.cam.stage_sigs["num_images"] = 1
        
    print("Setting up ROI and STATS defaults ...", end=" ")
    for name in simDet.component_names:
        if "roi" in name:
            roi = getattr(simDet, name)
            roi.wait_for_connection(timeout=10)
            roi.nd_array_port.put("SIM1")
        if "stats" in name:
            stat = getattr(simDet, name)
            stat.wait_for_connection(timeout=10)
            stat.nd_array_port.put(f"ROI{stat.port_name.get()[-1]}")

    print("Done!")
    print("Setting up defaults kinds ...", end=" ")
    simDet.default_kinds()
    print("Done!")
    print("Setting up default settings ...", end=" ")
    simDet.default_settings()
    print("Done!")
    print("All done!")
    
    return simDet


def loadShadowDet():
#   shadowDet = ShadowDetector()
    shadowDet = None
    
    return shadowDet
