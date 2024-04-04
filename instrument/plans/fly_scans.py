__all__ = """
    staged_fly
""".split()

from ..session_logs import logger

logger.info(__file__)

from bluesky import preprocessors as bpp
from bluesky import plan_stubs as bps
from bluesky import plans as bp


def staged_fly(flyers):
    
    @bpp.stage_decorator(flyers)
    def inner_fly():
        yield from bp.fly(flyers)
        
    return (yield from inner_fly())
