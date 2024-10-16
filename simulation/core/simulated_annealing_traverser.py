from simulation.core.traverser_base import *


class SimulatedAnnealingTraverser(TraverserBase):
    def __init__(self, system):
        super().__init__(system)
        self.__temperaturePoint = 0
        self.__temperatureStep = 0.01

    def __generateSequence(self):
        self._currentSequence = copy.deepcopy(self._bestSequence)
        self.__modifySequence()
        self._tmpSequence = copy.deepcopy(self._currentSequence)

    def __modifySequence(self):
        swaps = max(1, int(len(self._currentSequence) * 0.1))
        for i in range(0, swaps):
            pos1 = random.randint(0, len(self._currentSequence)-1)
            pos2 = random.randint(0, len(self._currentSequence) - 1)
            while pos1 == pos2:
                pos2 = random.randint(0, len(self._currentSequence)-1)
            self._currentSequence[pos1], self._currentSequence[pos2] = self._currentSequence[pos2], self._currentSequence[pos1]

    def nextIteration(self):
        self.__performStateTransition()
        self.__generateSequence()
        self.__temperaturePoint += self.__temperatureStep
        self._currentCost = 0
        self._currentStatistics = TraverserStatistics(0, 0, 0, 0)

    def feedback(self, cost, collisions, timeInQueue, timeInPenalty, timeInTransition):
        super().feedback(cost, collisions, timeInQueue, timeInPenalty, timeInTransition)

    def __performStateTransition(self):
        if self._bestCost == -1:
            self._acceptCurrentSolution()
        else:
            if random.random() < self.__transitionProbability(currentEnergy=self._bestCost, newEnergy=self._currentCost):
                self._acceptCurrentSolution()

    def __transitionProbability(self, currentEnergy, newEnergy):
        upwardsTransitionProbability = self.__temperature()
        if newEnergy > currentEnergy:
            return upwardsTransitionProbability
        return 1 - upwardsTransitionProbability

    def __temperatureFunction(self, x):
        f1 = math.exp(-x)
        f2 = math.cos(x*x)**2
        f3 = 1.5*x + 0.75
        f4 = 0.25
        return f1*f2*f3*f4

    def __temperature(self):
        return self.__temperatureFunction(self.__temperaturePoint)

