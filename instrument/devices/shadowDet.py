"""
synthetic detector based on shadow model of a beamline
"""
__all__ = """
    shadow0D
""".split()



from ..session_logs import logger

logger.info(__file__)

from ophyd.device import BlueskyInterface, Staged, Device, kind_context
from ophyd.signal import EpicsSignal, EpicsSignalRO, Kind, Signal
from ophyd.status import DeviceStatus
from ophyd.device import Component as Cpt
from .. import iconfig
    
IOC = iconfig.get("SB_IOC_PREFIX", "gp:")    
        
class SimpleTrigger(BlueskyInterface):
    """
    This trigger class is simple version modeled after quadEM but without the
    area detector related stuff.

    Examples
    --------

    >>> class SimDetector(SingleTrigger):
    ...     pass
    >>> det = SimDetector('..pv..')
    # optionally, customize name of image
    >>> det = SimDetector('..pv..', image_name='fast_detector_image')
    """
    
    _status_type = DeviceStatus

    def __init__(self, *args, image_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = None
        self._acquisition_signal = self.acquire
        
    def stage(self):
        self._acquisition_signal.subscribe(self._acquire_changed)
        super().stage()

    def unstage(self):
        super().unstage()
        self._acquisition_signal.clear_sub(self._acquire_changed)

    def trigger(self):
        "Trigger one acquisition."
        if self._staged != Staged.yes:
            raise RuntimeError(
                "This detector is not ready to trigger."
                "Call the stage() method before triggering."
            )

        self._status = self._status_type(self)
        self._acquisition_signal.put(1, wait=False)
        return self._status

    def _acquire_changed(self, value=None, old_value=None, **kwargs):
        "This is called when the 'acquire' signal changes."
        if self._status is None:
            return
        if (old_value == 1) and (value == 0):
            # Negative-going edge means an acquisition just finished.
            self._status.set_finished()
            self._status = None

class ShadowDetector0D(SimpleTrigger, Device):
    """
    stuff
    """
    with kind_context("omitted") as OCpt:
        acquire = OCpt(EpicsSignal, "Acquire")
    with kind_context("hinted") as HCpt:
        intensity = HCpt(EpicsSignalRO, "intensity")


#class ShadowDetector2D(SimpleTrigger,...):
#   """
#   stuff
#   """

# acquire PV 100idPySBL:SBL:det0d:Acquire
# data PV 100idPySBL:SBL:det0d:intensity
shadow0D = ShadowDetector0D(f"{IOC}SBL:det0d:", name="shadow0D", labels=("detector",))

#shadow2D = ShadowDetector2D
