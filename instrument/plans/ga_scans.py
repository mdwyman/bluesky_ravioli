__all__ = """
    ga_scan
""".split()

from ..session_logs import logger

logger.info(__file__)

import os
import inspect
from itertools import chain, zip_longest
from functools import partial
import collections
from collections import defaultdict
import time
import pyRestTable

from apstools.plans.doc_run import write_stream

import numpy as np

from bluesky import utils, plan_patterns
from bluesky.utils import Msg
import bluesky.plan_stubs as bps

from bluesky import preprocessors as bpp

from ophyd import Component
from ophyd import Device
from ophyd import Signal

from .ga4beamlines import GA4Beamline
from .ga4beamlines import sMode, pMode, cxMode, mMode

DEFAULT_PARAMETERS = {'survivor':   sMode[1],     # Genitor
                      'parent':     pMode[0],     # ProbRank 
                      'cx':         cxMode[1],    # Simple
                      'mx':         mMode[1]}     # Gaussian mutation

def bsFitnessFunc(self, pop):
    tmpFit = []
    def getFitness():
        for row in range(len(pop.index)):
            positions = pop.iloc[row, :len(self.motors)].tolist()
            mv_args = [z for z in chain.from_iterable(zip_longest(self.epics_motors, positions))]
            yield Msg("checkpoint")
            yield from bps.mv(*mv_args)
            yield Msg("create", None, name="primary")
            for det in self.epics_det:
                yield Msg("trigger", det, group="B")
            yield Msg("wait", None, "B")
            devices = tuple(utils.separate_devices(self.epics_det + self.epics_motors))
            for det in devices: 
                cur_det = yield Msg("read", det)
                if self.fitness in cur_det:
                    cur_fit = cur_det[self.fitness]['value']
            yield Msg("save")
            tmpFit.append(cur_fit)
        pop["fitness"] = tmpFit
    
    return (yield from getFitness())

GA4Beamline._FitnessFunc = bsFitnessFunc

class gaResults(Device):
    """
    Provides bps.read() as a Device

    .. index:: Bluesky Device; gaResults
    """

    generation = Component(Signal)
    # - - - - -
    ave_fitness = Component(Signal)
    max_fitness = Component(Signal)
    # max_fitness_positions --> # can I create a signal/component who's value is a list or dict of positions
    gastats_attrs = "ave_fitness max_fitness".split()

    def report(self, title=None):
        keys = self.gastats_attrs
        t = pyRestTable.Table()
        t.addLabel("key")
        t.addLabel("result")
        for key in keys:
            v = getattr(self, key).get()
            t.addRow((key, str(v)))
        if title is not None:
            print(title)
        print(t)

    def set_stats(self, stats):
        for key in self.gastats_attrs:
            v = getattr(stats, key)
            if v is None:  # None cannot be mapped into json
                v = "null"  # but "null" can replace it
            if key in ("crossings", "min", "max"):
                v = np.array(v)
            getattr(self, key).put(v)


def convert_motors(motors, ga_limits = None):
    """
    Convert list of motors to list of dictionaries used by ga4beamlines: 
    ('name', PV name/signal name for epics motors), upper limit ('hi'), 
    lower limit ('lo'), and sigma ('sigma')
    
    motors: list of signal names 
        list of motors to search
    ga_limits: list of [lo, hi] 
        limits for motors for ga search
    """
    ga_motor_list = []
    for i, m in enumerate(motors):
        if ga_limits is None:
            limits = [m.low_limit, m.high_limit]
        else:
            limits = ga_limits[i]
        sigma = abs(limits[1] - limits[0])/5
        ga_motor_list.append({'name':m.name, 'lo':min(limits), 'hi':max(limits), 'sigma':sigma})
    
    return ga_motor_list

def evolution_not_done(curr_gen, max_gen, curr_fitness, fitness_min):
    if fitness_min is None:
        not_finished = (curr_gen < max_gen)
    else:
        not_finished = (curr_gen < max_gen) and (curr_fitness < fitness_min)
    
    return not_finished

def ga_scan(detectors, motors, fitness_func = None, ga_parameters = DEFAULT_PARAMETERS, starting_population = None,
            population_n = 10, max_n_generations = 10, fitness_minimum = None, ga_limits = None, mv_ideal = False, md=None):

    """
    Scan and evolve population of positions with goal of maximizing a fitness
    function.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    motors : list
        list of any 'settable' objects (motor, temp controller, etc.) whose position
        in the list corresponds to a trial setting in each population member
    fitness_func : string
        name of detector for determining fitness for each member of population
    ga_parameters : dictionary
        dictionary of parameters for genetic algorithm
    starting_population : list
        list of trial settings
    max_n_generations : integer
        number of generations to evolve through if fitness_minimum is None
    fitness_minimum : float
        early stopping criteria for genetic algorithm
    ga_limits: list of [lo, hi] 
        limits for GA search; if None, then use motor limits
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.adaptive_scan`
    :func:`bluesky.plans.rel_adaptive_scan`
    """

    # Any checks?

    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name for motor in motors],
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motors': repr(motors),
                         'start_population': starting_population,
                         'ga_parameters': ga_parameters,
#                        'fitness_func': fitness_func.__name__,
                         'max_n_generations': max_n_generations,
                         'fitness_minimum': fitness_minimum},
           'plan_name': 'ga_scan',
           'hints': {},
           }
    _md.update(md or {})

    try:
#        dimensions = [(motor.hints['fields'], 'primary')]
        dimensions = [(motor.hints["fields"], "primary") for motor in motors]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    if fitness_func is None:
#        fitness_func = detectors[0].name
        fitness_func = detectors[0].name
#        print(fitness_func)
        
    ga_motors = convert_motors(motors, ga_limits)

    if starting_population is None:
        ga = GA4Beamline(ga_motors, ga_parameters['survivor'],ga_parameters['parent'],
                         ga_parameters['cx'],ga_parameters['mx'], fitness_func, 
                         epics_motors = motors, epics_det = detectors, 
                         nPop = population_n)
    else:
        ga = GA4Beamline(ga_motors, ga_parameters['survivor'],ga_parameters['parent'],
                         ga_parameters['cx'],ga_parameters['mx'], fitness_func, 
                         epics_motors = motors, epics_det = detectors, 
                         initPop = starting_population)

    @bpp.stage_decorator(detectors + motors)
    @bpp.run_decorator(md=_md)
    def soga_core():
        print(f"Beginning GA with initial population of {ga.nPop}")
        curr_gen = 0
        curr_obj = 0
        fitness_history = []
    
        stream_name = "gaStats"
        results = gaResults(name=stream_name)

        if os.environ.get("BLUESKY_PREDECLARE", False):
            yield from bps.declare_stream(*motors, *detectors, name="primary")
        while evolution_not_done(curr_gen, max_n_generations, curr_obj, fitness_minimum):
            if curr_gen == 0:
                yield from ga.FirstGeneration()
            else:
                yield from ga.NextGeneration()
            fitnesses = ga.population.fitness
            fitness_history.append(fitnesses)
            curr_obj = max(fitnesses)

            results.generation.put(curr_gen)
            results.max_fitness.put(curr_obj)
            results.ave_fitness.put(np.mean(fitnesses))
#            results.best_candidate.put()
            
            results.report(stream_name)

            #What do I hope to get with this?
            ga.stats.append(results)

            try:
                yield from write_stream(results, label=stream_name) 
            except ValueError as ex:
                separator = " " * 8 + "-" * 12
                print(separator)
                print(f"Error saving stream {stream_name}:\n{ex}")
                print(separato)  
            
            curr_gen += 1
            
        results.report(stream_name)
        if mv_ideal:
            positions = ga.population.iloc[0, :len(ga_motors)].tolist()
            print(f"Moving to ideal position: {positions}")
            mv_args = [z for z in chain.from_iterable(zip_longest(motors, positions))]
            yield Msg("checkpoint")
            yield from bps.mv(*mv_args)

    return (yield from soga_core())

