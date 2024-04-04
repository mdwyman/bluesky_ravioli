"""
example scaler
"""

__all__ = """
	synDiode
	""".split()
    # "scaler1",
    # "timebase",
    # "I0",
    # "scint",
    # "diode",

from ..session_logs import logger

logger.info(__file__)

from ophyd.scaler import ScalerCH
from ophyd import Kind
from ophyd.sim import SynSignalRO
from .motors import m3

def synIntensity():
	
	limits = [m3.get_lim(-1), m3.get_lim(1)]
	lo_limit = min(limits)
	hi_limit = max(limits)
	position = m3.user_readback.get()
	peak = 200.0
	
	return peak*(position - lo_limit)/(hi_limit - lo_limit)


synDiode = SynSignalRO(func=synIntensity, name="synDiode", labels=["detectors"])

# # make an instance of the entire scaler, for general control
# scaler1 = ScalerCH("ioc:scaler1", name="scaler1", labels=["scalers", "detectors"])

# # choose just the channels with EPICS names
# scaler1.select_channels()

# # examples: make shortcuts to specific channels assigned in EPICS

# timebase = scaler1.channels.chan01.s
# I0 = scaler1.channels.chan02.s
# scint = scaler1.channels.chan03.s
# diode = scaler1.channels.chan04.s

# for item in (timebase, I0, scint, diode):
#     item._ophyd_labels_ = set(["channel", "counter",])
