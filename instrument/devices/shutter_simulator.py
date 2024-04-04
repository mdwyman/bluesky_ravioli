"""
shutter
"""

__all__ = """
    shutterA
    shutterB
    shutter
""".split()

from ..session_logs import logger

logger.info(__file__)

from ophyd import Device, EpicsSignal, EpicsSignalRO, Signal, SignalRO
from ophyd import Component as Cpt
from ophyd.status import DeviceStatus
from apstools.devices import ShutterBase, OneSignalShutter, SimulatedApsPssShutterWithStatus
import apstools.devices
import time



class SoftShutter(ShutterBase):

    setpoint = Cpt(EpicsSignal, ".C")
    readback = Cpt(EpicsSignalRO, ".VAL")
    
    def open(self):
        if not self.isOpen:
            self.setpoint.put(self.open_value)
            if self.delay_s > 0:
                time.sleep(self.delay_s)    # blocking call OK here
        
    def close(self):
        if not self.isClosed:
            self.setpoint.put(self.close_value)
            if self.delay_s > 0:
                time.sleep(self.delay_s)    # blocking call OK here
       
    def state(self):
        if self.readback.get() == self.open_value:
            result = self.valid_open_values[0]
        elif self.readback.get() == self.close_value:
            result = self.valid_close_values[0]
        else:
            result = self.unknown_state
        return result



shutterA = SoftShutter('100idWYM:userCalcOut9', name = 'shutterA')
shutterB = SoftShutter('100idWYM:userCalcOut10', name = 'shutterB')

shutter = SimulatedApsPssShutterWithStatus(name="shutter", labels=("shutters",))

# shutter needs short recovery time after moving
shutter.delay_s = 0.05
