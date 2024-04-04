from ophyd import Device, EpicsSignal, EpicsSignalRO, Component
from ophyd import Component as Cpt
from ophyd.status import DeviceStatus
import time
import logging

logger = logging.getLogger()

__all__ = """
    testFlyer
""".split()


'''
PUt in devices and will want to add to init: "from 2d_flyers import *"
'''

# fly scan usercalc enable
#flyUserCalcEnable = EpicsSignal('2iddVELO:userCalcEnable.VAL')
# step scan usercalc enable
#scanUserCalcEnable = EpicsSignal('2iddf:userCalcEnable.VAL')

# Laser frequency
#laserFrequency = EpicsSignal('2iddVELO:afg:set_freq')

'''
Using https://github.com/APS-2BM-MIC/ipython-user2bmb/blob/e0e601c84f41163ce3f08d88cb24b7e66a0ca65c/profile_2bmb/startup/taxi_and_fly.ipynb
as template
'''

monitor_PV = '100idWYM:scan1.EXSC'
mode_PV = '100idWYM:scan1.P1SM'
npts_PV = '100idWYM:scan1.NPTS'
trigger_PV = '100idWYM:scan1.EXSC'
acquire_PV = '100idWYM:simDet:cam1:Acquire'
camMode_PV = '100idWYM:simDet:cam1:ImageMode'
pos_readback_PV = '100idWYM:scan1.P1RA'
t_readback_PV = '100idWYM:scan1.P4RA'
pts_readback_PV = '100idWYM:scan1.CPTS'

class scanRecFlyer(Device):
    
    
    ''''
    monitor = Component(EpicsSignalRO,monitor_PV, name = 'scanMonitor') # != 1 not scanning, == 1 scanning
    start_program = Component(EpicsSignal, trigger_PV, name = 'scanTrigger') # set to 1 to start
    mode = Component(EpicsSignal, mode_PV, name = 'scanMode') # generally set to 5 for snake scans
    cam_acquire = Component(EpicsSignal, acquire_PV, name = 'camTrigger')
    camMode = Component(EpicsSignal, camMode_PV, name = 'camMode')
    '''

    def __init__(self, monitor_PV = '100idWYM:scan1.EXSC',
                  mode_PV = '100idWYM:scan1.P1SM',
                  npts_PV = '100idWYM:scan1.NPTS',
                  pos_readback_PV = '100idWYM:scan1.P1RA',
                  tim_readback_PV = '100idWYM:scan1.P4RA',
                  trigger_PV = '100idWYM:scan1.EXSC',
                  acquire_PV = '100idWYM:simDet:cam1:Acquire',
                  camMode_PV = '100idWYM:simDet:cam1:ImageMode',
                  *args, **kwargs ):

        super().__init__('', parent=None, *args, **kwargs)
        self.complete_status = None

        
        self.monitor_PV = monitor_PV
        self.mode_PV = mode_PV
        self.npts_PV = npts_PV
        self.pos_readback_PV = pos_readback_PV
        self.tim_readback_PV = tim_readback_PV
        self.trigger_PV = trigger_PV
        self.acquire_PV = acquire_PV
        self.camMode_PV = camMode_PV
        self.scanPt_PV = pts_readback_PV
        
        '''
        self.monitor = EpicsSignalRO(self.monitor_PV, name = 'scanMonitor') # != 1 not scanning, == 1 scanning
        self.start_program = EpicsSignal(self.trigger_PV, name = 'scanTrigger') # set to 1 to start
        self.mode = EpicsSignal(self.mode_PV, name = 'scanMode') # generally set to 5 for snake scans
        self.cam_acquire = EpicsSignal(self.acquire_PV, name = 'camTrigger')
        self.camMode = EpicsSignal(self.camMode_PV, name = 'camMode')
        '''
       
        '''
        self.monitor = Component(EpicsSignalRO,monitor_PV, name = 'scanMonitor') # != 1 not scanning, == 1 scanning
        self.start_program = Component(EpicsSignal, trigger_PV, name = 'scanTrigger') # set to 1 to start
        self.mode = Component(EpicsSignal, mode_PV, name = 'scanMode') # generally set to 5 for snake scans
        self.cam_acquire = Component(EpicsSignal, acquire_PV, name = 'camTrigger')
        self.camMode = Component(EpicsSignal, camMode_PV, name = 'camMode')
        '''
        self.monitor = EpicsSignalRO(self.monitor_PV, name = 'scanMonitor') # != 1 not scanning, == 1 scanning
        self.start_program = EpicsSignal(self.trigger_PV, name = 'scanTrigger') # set to 1 to start
        self.npts = EpicsSignal(self.npts_PV, name = 'number_pts') # set to 1 to start
        self.positions = EpicsSignalRO(self.pos_readback_PV, name = 'scanPositions') # != 1 not scanning, == 1 scanning
        self.times = EpicsSignalRO(self.tim_readback_PV, name = 'scanTimes') # != 1 not scanning, == 1 scanning
        self.scanPt = EpicsSignalRO(self.scanPt_PV, name = 'scanPt') # != 1 not scanning, == 1 scanning
        self.mode = EpicsSignal(self.mode_PV, name = 'scanMode') # generally set to 5 for snake scans
        self.cam_acquire = EpicsSignal(self.acquire_PV, name = 'camTrigger')
        self.camMode = EpicsSignal(self.camMode_PV, name = 'camMode')
        
        self.t0 = 0
        self.t1 = 0
        
        self.stage_sigs['mode'] = 2 # Fly for scan record
#       self.stage_sigs[acquire_PV] = 1 
        self.stage_sigs['camMode'] = 0 # single acquisition mode
#       self.stage_sigs[self.user_offset] = 5

    def stage(self):
        super().stage()
        print('Flyer staged.')
 
    def unstage(self):
        super().unstage()

        self.cam_acquire.unsubscribe(self.acquire_cb)
        self.monitor.unsubscribe(self.monitor_cb)

        print('Flyer unstaged.')

    def kickoff(self):
            """
            Start this Flyer
             """
            logger.info("kickoff()")
            self.complete_status = DeviceStatus(self)

            #trigger system
            #self.busy.put(BusyStatus.busy) -- from example
            #send trigger to start_program
            self.start_program.put(1)
            while(self.monitor.get() != 1):
                print("waiting for scan Record to begin")
                time.sleep(0.01)
            
            #send trigger to camera
            #self.cam_acquire.put(1)

            #add callback functions to set complete after fly scan trajectory
            #and detector acquisition complete
            def cb(*args, **kwargs):
                if not self.monitor.get() and not self.cam_acquire.get():
                    self.complete_status._finished(success=True)
                    print('\n Fly scan callback triggered.')
                    #self.complete_status.set_finished()
 
            self.monitor_cb = self.monitor.subscribe(cb)
            self.acquire_cb = self.cam_acquire.subscribe(cb)

            self.t0 = time.time()
            
            # set kickoff status to done
            kickoff_status = DeviceStatus(self)
            kickoff_status._finished(success=True)
            return kickoff_status

    def complete(self):
        """
        Wait for flying to be complete
        """
        logger.info("complete(): " + str(self.complete_status))
        print('Complete function run: '+str(self.complete_status))
        if self.complete_status.success:
            print('Unsubscribing callbacks {} and {}.'.format(self.acquire_cb, self.monitor_cb))
            # Moved the unsubscribes because the complete function wasn't being called
#            self.cam_acquire.unsubscribe(self.acquire_cb)
#            self.monitor.unsubscribe(self.monitor_cb)
            print("Fly scan completed")

        return self.complete_status

    def describe_collect(self):
        """
        Describe details for ``collect()`` method
        """
        logger.info("describe_collect()")
        return {
            self.name: dict(
                ifly_xArr = dict(
                    source = self.positions.pvname,
                    dtype = "number",
                    shape = (1,)
                ),
                ifly_tArr = dict(
                    source = self.times.pvname,
                    dtype = "number",
                    shape = (1,)
                )
            )
        }

    def collect(self):
        """
        Start this Flyer
        """
        logger.info("collect(): " + str(self.complete_status))
        print("Collecting data")
        self.t1 = time.time()
        
        # x, t are the possitions and times recorded by the scan record
#        t = self.times.value
#        x = self.positions.value
        
        # testing a simpler dataset: start time, end time, number of points
        t = [0, self.t1 - self.t0]
        x = [0, self.npts.value]
        
        # In general, put d assignment in for loop with yield within said loop
        # at end.  Collect will then yield a single time point + whatever data
        # is wanted at that point.  Later on should use flyer's position data?
#        for i in range(self.npts.value):
        for i in range(2):
            d = dict(
                time=self.t0 + t[i],
                data=dict(
                    ifly_tArr = t[i],
                    ifly_xArr = x[i],
                ),
                timestamps=dict(
                    ifly_tArr = t[i],
                    ifly_xArr = t[i],
                )
            )
            yield d



scanRec_monitor_PV = '100idWYM:scan1.EXSC'
scanRec_mode_PV = '100idWYM:scan1.P1SM'
scanRec_trigger_PV = '100idWYM:scan1.EXSC'
scanRec_acquisition_PV = '100idWYM:simDet:cam1:Acquire'
scanRec_camMode_PV = '100idWYM:simDet:cam1:ImageMode'

testFlyer = scanRecFlyer(monitor_PV = scanRec_monitor_PV,
                         mode_PV = scanRec_mode_PV,
                         trigger_PV = scanRec_trigger_PV,
                         acquire_PV = scanRec_acquisition_PV,
                         camMode_PV = scanRec_camMode_PV,
                         name = 'testFlyer')
