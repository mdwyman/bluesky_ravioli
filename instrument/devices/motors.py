"""
Using motors from simulated (Shadow) beamline
"""

__all__ = """
    m1  m2  m3  m4
    m5  m6  m7  m8
    m9  m10 m11 m12
""".split()

from ..session_logs import logger

logger.info(__file__)

from .. import iconfig
from ophyd import EpicsMotor, Component, EpicsSignal


#IOC = iconfig.get("GP_IOC_PREFIX", "gp:")
IOC = iconfig.get("SB_IOC_PREFIX", "gp:")


class MyEpicsMotor(EpicsMotor):
    steps_per_revolution = Component(EpicsSignal, ".SREV", kind="omitted")


m1 = MyEpicsMotor(f"{IOC}m1", name="m1", labels=("motor",))
m2 = MyEpicsMotor(f"{IOC}m2", name="m2", labels=("motor",))
m3 = MyEpicsMotor(f"{IOC}m3", name="m3", labels=("motor",))
m4 = MyEpicsMotor(f"{IOC}m4", name="m4", labels=("motor",))
m5 = MyEpicsMotor(f"{IOC}m5", name="m5", labels=("motor",))
m6 = MyEpicsMotor(f"{IOC}m6", name="m6", labels=("motor",))
m7 = MyEpicsMotor(f"{IOC}m7", name="m7", labels=("motor",))
m8 = MyEpicsMotor(f"{IOC}m8", name="m8", labels=("motor",))
m9 = MyEpicsMotor(f"{IOC}m9", name="m9", labels=("motor",))
m10 = MyEpicsMotor(f"{IOC}m10", name="m10", labels=("motor",))
m11 = MyEpicsMotor(f"{IOC}m11", name="m11", labels=("motor",))
m12 = MyEpicsMotor(f"{IOC}m12", name="m12", labels=("motor",))
#m13 = MyEpicsMotor(f"{IOC}m13", name="m13", labels=("motor",))
#m14 = MyEpicsMotor(f"{IOC}m14", name="m14", labels=("motor",))
#m15 = MyEpicsMotor(f"{IOC}m15", name="m15", labels=("motor",))
#m16 = MyEpicsMotor(f"{IOC}m16", name="m16", labels=("motor",))

m1.wait_for_connection()
m1.steps_per_revolution.put(200)
