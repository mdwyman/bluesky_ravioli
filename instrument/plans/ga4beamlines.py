#!/usr/bin/env python 3

import pandas as pd
import random
import numpy as np
#import ackley
from scipy.stats import truncnorm

#################### VARIABLES AND CONSTANTS DEFINITIONS ####################
#Valid survivor selection methods
sMode =     [{"name": "age", "nElite": 0},
             {"name": "genitor"},
             {"name": "steady", "nChild": 2}]
#Valid parent selection methods
pMode =     [{"name": "probRank", "s": 1.5},
             {"name": "probFit"}]
#Valid child generation methods
cxMode =    [{"name": "single", "alpha": 0.5},
             {"name": "simple", "alpha": 0.75},
             {"name": "whole", "alpha": 0.75}]
#Valid mutation methods
mMode =     [{"name": "uniform"},
             {"name": "gaussian"},
             {"name": "cauchy"}]
#Valid fitness methods
#fMode =     [{"type": "Func", "name": ackley.AckleyFunc}]

#################### CLASS DEFINITIONS ####################
########## ERROR CLASSES ##########

class BeamlineError(Exception):
    '''This is the base class for exeptions in this module'''
    pass

class MethodError(BeamlineError):
    '''Exception raised for errors in methods to use'''

    def __init__(self, message):
        super().__init__(message)

########## GA CLASS ##########

class GA4Beamline():
    """
    Attempts to find a satisfactory motor configuration for a beamline using genetic algorithms using specified methods.

    ...

    Parameters
    ----------
    motors : list of dict
        Contains dictionaries for each motor's transformations with their name ('name', PV name for epics motors),
            upper limit ('hi'), lower limit ('lo'), and sigma ('sigma').
    survivorMode : dict
        Specifies the survivor selection method ('name') and parameters ('nElite', optional).  See sMode for valid parameters.
    parentMode : dict
        Parent selection method ('name') and parameters ('s', optional).  See pMode for valid parameters.
    cxMode : dict
        Recombination method ('name') and parameters (kwargs: 'alpha').  See cxMode for valid parameters.
    mutationMode : dict
        Mutation method ('name').  See mMode for valid parameters.
    fitness : dict
        How to measure fitness. 'Type' is either ‘PV’ or ‘Func’ and 'name' is the either the PV or function name to be used.
            See fMode for valid parameters.
    nPop : int, optional
        Number of individuals in the population (Default value is 10).
    initPop : pandas data frame, optional
        Initial population; if equal to None, will create one (Default value is None).  Should have a column for each of the
            motor names in motors as well as a 'fitness', 'rank', and 'probability' column in that order.  Should have an
            index length equal to nPop.
    OM : bool, optional
        Turns on Observer Mode – only set to True when using against epics motors/fitness function (Default value is False).

    Attributes
    ----------
    motors : list of dict
        Stores the value of motors passed in to initialize the class.
    generation : int
        Current generation of the population.
    sSel : dict
        The method ('name') and parameters ('nElite') to use for survivor selection.
    pSel : dict
        The method ('name') and parameters ('s') to use for parent selection.
    cxMode : dict
        The method ('name') and parameters ('alpha') to use for recombination.
    mMode : dict
        The method ('name') to use for recombination.
    obsMode : bool
        Determines whether to use oberver mode (True) or not.  Should use only when using epics motors/fitness function.
    fitness : dict
        The type of fitness function to use ('type') and the function name ('name') to use for fitness evaluation.
    nPop: int
        The number of individuals in the population each generation.
    population: pandas data frame
        Stores the current generation of motor configurations.  Has columns for each motor, overall fitness of individual,
            the rank of the individual, and the probability of selecting it as a parent.
    parents : list of indexes
        The indexes of the individuals in the population to use in child generation.
    children : pandas data frame
        Stores the potential next generation of motor configurations.  Has columns for each motor, overall fitness of individual,
            the rank of the individual, and the probability of selecting it as a parent.
    fitHistory : pandas data frame
        Stores the average average fitness, peak fitness, and peak motor configuration for each generation.

    Methods
    -------
    FirstGeneration()
        Finishes setting up and evaluating population and prepares the algorithm for multiple generations.  Must be called before NextGeneration().
    NextGeneration()
        Progresses the algorithm forward to the next generation.  Continual iteration (and therefore termination) must be handled externally.

    """

    def __init__(self, motors, survivorMode, parentMode, cxMode, mutationMode,
                    fitness, nPop = 10, initPop = None, OM = False, epics_motors = None,
                    epics_det = None):

        self.motors = motors
        self.motorNames = [motor['name'] for motor in motors]
        self.generation = 0
        self.nPop = nPop
        self.sSel = self._VerifySurvivorMode(survivorMode)
        self.pSel = self._VerifyParentMode(parentMode)
        self.cxMode = self._VerifyCXMode(cxMode)
        self.mMode = self._VerifyMMode(mutationMode)
        self.obsMode = OM
        self.fitness = fitness
        self.epics_motors = epics_motors
        self.epics_det = epics_det
        if initPop is None:
            self.population = self._CreatePop()
        else:
            self.population = initPop

        self.stats = []
        self.parents = []
        self.children = pd.DataFrame(self._MakeDataFrameCat())
        self.fitHistory = pd.DataFrame({"aveFitness": [], "peakFitness": [], "peakParameters": []})

    def _CreatePop(self):
        """
        Initializes population if none was provided.
        """
        categories = {}

        for motor in self.motors:
            categories[motor["name"]] = []

            for i in range(self.nPop):
                categories[motor["name"]].append(random.uniform(motor["lo"], motor["hi"]))

        categories["fitness"] = np.zeros(self.nPop)
        categories["ranking"] = [0] * self.nPop
        categories["probability"] = np.zeros(self.nPop)

        population = pd.DataFrame(categories)

        return population

    def FirstGeneration(self):
        """
        Primes the algorithm.  MUST be run before NextGeneration().
        """
        print(f"Starting first generation")
        yield from self._Measure(childrenOnly = False)
        self._SurvivorSel()
        

    def NextGeneration(self):
        """
        Progresses the algorithm.  NOTE: Continual calls and termination must be handled externally.
        """
        print(f"Starting a new generation.")
        self._ParentSel()
        self._Recombine()
        self._Mutate()
        yield from self._Measure()
        self._SurvivorSel()
        
    #################### STAGES FUNCTIONS ####################

    def _SurvivorSel(self):
        """
        Determines which individuals are carried over into the next generation.
        """
        tmp = None

        if self.generation == 0:
            self.generation += 1
        else:
            self.generation += 1

            if self.sSel["name"] == "age":
                if self.sSel["nElite"] > 0:
                    self.population = self.population.iloc[0:self.sSel["nElite"], :]
                    #print(f"Copied elite.  public is:\n{self.population}")

                    self.population = pd.concat([self.population, self.children.iloc[:(self.nPop - self.sSel["nElite"]), :]], ignore_index = True)
                    #print(f"Combined old and new.  public is:\n{self.population}")
                else:
                    self.population = self.children.iloc[:self.nPop, :]

            elif self.sSel["name"] == "genitor":
                self.population = pd.concat([self.population, self.children], ignore_index = True)

                #Use rankPop to set rank column of population + children
                self._RankPop()

                self.population = self.population.iloc[0:self.nPop, :]

        #Update fitHistory
        tmp = pd.DataFrame({"aveFitness": [self.population["fitness"].mean()],
                            "peakFitness": [self.population["fitness"].max()],
                            "peakParameters": [self.population.iloc[0, : len(self.motors)].tolist()]})
        self.fitHistory = pd.concat([self.fitHistory, tmp], ignore_index = True)

    def _ParentSel(self):
        """
        Selects individuals from the population to use to generate new individuals for the next generation.
        """
        #Use rankPop to set rank column
        self._RankPop()

        if self.pSel["name"] == "probRank":
            self._CalcProb("rank")

        elif self.pSel["name"] == "probFit":
            self._CalcProb("fitness")

        self.parents = self._StochasticUnivSampling(numParents = self.nPop - self.sSel["nElite"])

        #print(f"parents has a length of: {len(self.parents)} and is:\n{self.parents}")

    def _StochasticUnivSampling(self, numParents):
        """
        Selects parents from population using a Stochastic Universal Sampling algorithm.

        Parameters
        ----------
        numParents : int
            The number of individuals to add to the parent pool.
        """
        cmlProb = self.population["probability"].cumsum().tolist()
        parents = []

        #print(cmlProb)

        currMember = i = 0
        r = random.uniform(0, 1 / numParents)

        #print(f"r is: {r}")

        while currMember < numParents:
            while r <= cmlProb[i] and currMember < numParents:
                parents.append(i)
                r += 1 / numParents
                #print(f"r is: {r}")
                currMember += 1

            i += 1

        #print(f"The parents are:\n{parents}")
        return parents


    def _Recombine(self):
        """
        Generates new motor configurations ('children') from the individuals in parents.
        """
        self.children = pd.DataFrame(self._MakeDataFrameCat())
        #print(f"children is:\n{self.children}")

        pairs = self._CreatePairs(self.parents)

        for p in range(len(pairs)):
            self.children = pd.concat([self.children, self._Recombination(pairs[p], self.cxMode)], ignore_index = True)
            #print([self.children, self.Recombination(pairs[p], self.cxMode)])
        #print(f"\nchildren is:\n{self.children}")

    def _CreatePairs(self, parents):
        """
        Creates a list of randomly selected pairs of parents.

        Parameters
        ----------
        parents : list of indexes
            The pool of potential parents in population.
        """
        pairs = []

        for i in range(int(np.ceil(len(parents) / 2))):
            pairs.append(random.sample(parents, 2))

        #print(f"The pairs are:\n{pairs}")

        return pairs

    def _Recombination(self, parents, mode):
        """
        Generates 2 children from each pair of parents using the method specified in mode.

        Parameters
        ----------
        parents : list of indexes
            Contains 2 individuals from population that will be used to generate new motor configurations.
        mode : dict
            Specifies the method and parameters to use to generate the new motor configurations.

        """
        alpha = mode["alpha"]
        parent1 = self.population.iloc[parents[0], :].tolist()
        parent2 = self.population.iloc[parents[1], :].tolist()
        child1 = None
        child2 = None

        #print(f"parent1 is: {parent1}\nparent2 is: {parent2}\n")

        #pick a random allele (k)
        #NOTE: DOING THIS SINCE THE MOTORS ARE THE FIRST COLUMNS IN THE POPULATION DATATABLE.  IF THIS IS CHANGED,
        #       THEN THE METHOD TO GENERATE K NEEDS TO CHANGE.
        k = int(random.choice(range(len(self.motors))))
        #print(f"k is: {k}")

        if mode["name"] == "single":
            #print(f"Parent1[k] is: {parent1[k]}\nParent2[k] is: {parent2[k]}")
            child1 = parent1[0:k]
            child1.append(parent1[k] * (1.0 - alpha) + parent2[k] * alpha)
            child1 = child1 + parent1[k + 1:]

            child2 = parent2[0:k]
            child2.append(parent2[k] * (1.0 - alpha) + parent1[k] * alpha)
            child2 = child2 + parent2[k + 1:]

        elif mode["name"] == "simple":
            #print(f"Parent1[k:] is: {parent1[k:]}\nParent2[k:] is: {parent2[k:]}")
            child1 = parent1[0:k]
            child1 = child1 + np.add(np.multiply(parent1[k:], 1 - alpha), np.multiply(parent2[k:], alpha)).tolist()

            child2 = parent2[0:k]
            child2 = child2 + np.add(np.multiply(parent2[k:], 1 - alpha), np.multiply(parent1[k:], alpha)).tolist()

        elif mode["name"] == "whole":
            child1 = np.add(np.multiply(parent1, 1 - alpha), np.multiply(parent2, alpha)).tolist()

            child2 = np.add(np.multiply(parent2, 1 - alpha), np.multiply(parent1, alpha)).tolist()

        #print(f"{child1}\n\n")

        tmp = self._MakeDataFrameCat()
        i = 0

        for column in tmp:
            tmp[column].append(child1[i])
            tmp[column].append(child2[i])
            i += 1

        tmp = pd.DataFrame(tmp)
        #print(f"\ntmp is:\n{tmp}")

        return tmp


    def _Mutate(self):
        """
        Causes changes in the values of the individuals in children based on the method specified in mMode.
        """
        for row in self.children.index:
            child = self.children.loc[row, :].tolist()
            self.children.loc[row, :] = self._Mutation(child, self.motors, self.mMode["name"])

    def _Mutation(self, child, motors, mode):
        """
        Makes changes in the values of the child based on the method specified in mMode.

        Parameters
        ----------
        child : list
            The values of a particular row in the children data frame converted into a list.
        motors : list of dict

        """
        #print(f"Before mutation, child is:\n{child}")

        mutatedValue = []

        for i in range(len(motors)):
            #print(f"Motor is: {motors[i]['name']}\nRange is: ({motors[i]['lo']}, {motors[i]['hi']})")
            #print(f"Original value is: {child[i]}")
            if mode == "gaussian":
                #Set low end in terms of standard deviations from current value
                a = (motors[i]['lo'] - child[i])/motors[i]['sigma']
                #Set high end in terms of standard deviations from current value
                b = (motors[i]['hi'] - child[i])/motors[i]['sigma']

                mutatedValue.append(truncnorm.rvs(a, b, loc = child[i], scale = motors[i]['sigma'], size=1)[0])

            elif mode == "uniform":
                mutatedValue.append(random.uniform(motors[i]["lo"], motors[i]["hi"]))
            #print(f"New mutated value is: {mutatedValue[i]}")

        mutatedValue = mutatedValue + child[len(motors):]
        #print(f"After mutation, child is:\n{mutatedValue}\n")

        return mutatedValue

    def _FitnessFunc(self, pop):
        print('Old fitness function running')
        #NOTE: WILL FINISH LATER
        tmpFit = []

        if self.fitness["type"] == "epics":
            pass
            '''
            will implement code later but it would look like the following:

            for p in population: # does for loop create unattached copy of range values?
                move motors to p[motor] values
                if not OM:
                    wait for motor move to complete
                else:# implement Observer mode

                 p[‘fitness’] = read fitness[‘pv’]
            '''
        elif self.fitness["type"] == "Func":
            try:
                fargs = self.fitness["args"]
            except:
                fargs = None
            for row in range(len(pop.index)):
                value = pop.iloc[row, :len(self.motors)].tolist()
                value = self.fitness["name"](value, fargs)
                tmpFit.append(value)

            #print(f"tmpFit is:\n{tmpFit}")
            pop["fitness"] = tmpFit
            #print(f"\npopulation is:\n{self.population}")

        #print(pop)
        #returns population but with the fitness values filled in

    def _RankPop(self):
        #NOTE: Need to verify functionality

        #Set the ranking column of the self.population dataframe (1 being the highest for descending = True)
        self.population = self.population.sort_values(by = ["fitness"], ascending = False)
        self.population["ranking"] = [i for i in range(len(self.population.index) - 1, -1, -1)]
        self.population.index = [i for i in range(len(self.population.index))]
        #print(self.population)

    def _CalcProb(self, probMode):
        #NOTE: WILL FINISH LATER.  Need to implement "rank"
        probs = []

        #Set the probability column in the self.population dataframe
        if probMode == "rank":
            #Loop through population and set probability using RankingProb
            for row in self.population.index:
                probs.append(self._RankingProb(self.population.loc[row, "ranking"], self.nPop, self.pSel['s']))

        elif probMode == "fitness":
            #Get sum of Fitness
            cmltFitness = self.population["fitness"].sum()
            #print(f"cumulative fitness is: {cmltFitness}")
            #Loop through population and set probability column to individual fitness/cumulative fitness
            for row in self.population.index:
                probs.append(self.population.loc[row, "fitness"] / cmltFitness)

        #print(f"probs sum is: {np.sum(probs)}")

        self.population["probability"] = probs
        #print(self.population)


    def _RankingProb(self, rank, nPop, s):
        return (2 - s) / nPop + 2 * rank * (s - 1) / nPop / (nPop - 1)

    def _Measure(self, childrenOnly = True):
#        print(f"Measuring fitness")
        #NOTE: Need to verify functionality

        #SINCE FITNESSFUNC IS SUPPOSED TO RETURN A MODIFIED POPULATION, I SHOULD PROBABLY SET SOMETHING EQUAL TO THESE
        if childrenOnly:
#            print(f"Measuring children")
            yield from self._FitnessFunc(self.children)
            self.children = self.children.sort_values(by = ["fitness"], ascending = False)
            self.children.index = [i for i in range(len(self.children.index))]

        else:
#            print(f"Measuring population")
            yield from self._FitnessFunc(self.population)

    #################### INITIALIZATION HELPER FUNCTIONS ####################

    def _VerifySurvivorMode(self, survivorMode):
        '''
        # Purpose:
            Ensure that valid values have been passed in for determining the survivor selection method.

        # Parameters:
            # survivorMode  : Survivor selection method (name: Age, progRank, probFit) and parameters (nElite optional)
        '''
        valid = False
        tmpDict = {}

        for dictn in sMode:
            if dictn["name"] == survivorMode["name"]:
                valid = True
                tmpDict["name"] = survivorMode["name"]

                if "nElite" in survivorMode:
                    if 0 <= survivorMode["nElite"] and survivorMode["nElite"] < self.nPop:
                        tmpDict["nElite"] = survivorMode['nElite']
                    else:
                        raise ValueError(f"{survivorMode['nElite']} is out of range [0, population size).")

                elif "nChild" in survivorMode:
                    if 0 < survivorMode["nChild"] and survivorMode["nChild"] < self.nPop:
                        tmpDict["nElite"] = self.nPop - survivorMode['nChild']
                    else:
                        raise ValueError(f"{survivorMode['nChild']} is out of range (0, population size).")

                else:
                    tmpDict["nElite"] = 0

                break

        if not valid:
            raise MethodError(message = f"{survivorMode['name']} is not a valid method for Survivor Selection.")

        return tmpDict

    def _VerifyParentMode(self, parentMode):
        '''
        # Purpose:
            Ensure that valid values have been passed in for determining the parent selection method.

        # Parameters:
            # parentMode    : Parent selection method (name: Fitness) and parameters (alpha)
        '''
        valid = False
        tmpDict = {}

        for dictn in pMode:
            if dictn["name"] == parentMode["name"]:
                valid = True
                tmpDict["name"] = parentMode["name"]

                #NOT SURE IF THERE SHOULD BE ANY CHECKS ON THE VALUE OF "S" OR WHAT SHOULD HAPPEN IF IT ISN'T DEFINED
                if "s" in parentMode:
                    if 1.0 < parentMode['s'] and parentMode['s'] <= 2.0:
                        tmpDict['s'] = parentMode['s']
                    else:
                        raise ValueError(f"{parentMode['s']} is not a valid 's' value")
                else:
                    tmpDict['s'] = dictn['s']

                break

        if not valid:
            raise MethodError(message = f"{parentMode['name']} is not a valid method for Parent Selection.")

        return tmpDict

    def _VerifyCXMode(self, childMode):
        '''
        # Purpose:
            Ensure that valid values have been passed in for determining the recombination method.

        # Parameters:
            # cxMode        : Recombination method (name: simple,single or whole) and parameters (kwargs: alpha)
        '''
        valid = False
        tmpDict = {}

        for dictn in cxMode:
            if dictn["name"] == childMode["name"]:
                valid = True
                tmpDict["name"] = childMode["name"]

                if "alpha" in childMode:
                    if 0.0 <= childMode["alpha"] and childMode["alpha"] <= 1.0:
                        tmpDict["alpha"] = childMode['alpha']
                    else:
                        raise ValueError(f"{childMode['alpha']} is not a valid 'alpha' value.")
                else:
                    tmpDict["alpha"] = dictn["alpha"]

                break

        if not valid:
            raise MethodError(message = f"{childMode['name']} is not a valid method for Recombination.")

        return tmpDict

    def _VerifyMMode(self, mutationMode):
        '''
        # Purpose:
            Ensure that valid values have been passed in for determining the mutation method.

        # Parameters:
            # cxMode        : Recombination method (name: simple,single or whole) and parameters (kwargs: alpha)
        '''
        valid = False
        tmpDict = {}

        for dictn in mMode:
            if dictn["name"] == mutationMode["name"]:
                valid = True
                tmpDict["name"] = mutationMode["name"]

                break

        if not valid:
            raise MethodError(message = f"{mutationMode['name']} is not a valid method for Mutation.")

        return tmpDict

    def _MakeDataFrameCat(self):
        categories = {}

        for motor in self.motors:
            categories[motor["name"]] = []

        categories["fitness"] = []
        categories["ranking"] = []
        categories["probability"] = []

        #tmp = pd.DataFrame(categories)

        return categories
