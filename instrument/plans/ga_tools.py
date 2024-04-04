import itertools
from .ga4beamlines import GA4Beamline
from .ga4beamlines import sMode, pMode, cxMode, mMode
import bluesky.plan_stubs as bps
import bluesky.plans as bp 

survivorModes = sMode
parentModes = pMode
childModes = cxMode
mutationModes = mMode

class bsga(GA4Beamline):
        
    def _FitnessFunc(self, pop):
        print('New fitness function running')
        tmpFit = []
        for p in pop:
            positions = p.iloc[row, :len(self.motors)].tolist()
            # another option for the mv_args:
            #[x for y in zip(motors, positions) for x in y]
            mv_args = [z for z in itertools.chain.from_iterable(itertools.zip_longest(self.motors, positions))]
            yield Msg("checkpoint")
            yield from bps.mv(*mv_args)
            yield Msg("create", None, name="primary")
            for det in detectors:
                yield Msg("trigger", det, group="B")
            yield Msg("wait", None, "B")
            for det in devices:
                cur_det = yield Msg("read", det)
                if self.fitness in cur_det:
                    cur_fit = cur_det[self.fitness]["value"]
            yield Msg("save")
            tmpFit.append(cur_fit)
        pop["fitness"] = tmpFit



