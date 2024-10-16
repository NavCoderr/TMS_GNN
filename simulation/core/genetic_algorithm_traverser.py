from simulation.core.traverser_base import *
from simulation.core.task import Task


DEFAULT_POOL_SIZE = 120
MUTATION_PROBABILITY = 0.15


def sequenceHistogram(sequence):
    res = {}
    for task in sequence:
        if task.taskNumber() in res:
            res[task.taskNumber()] += 1
        else:
            res[task.taskNumber()] = 1
    return res


class Genome:
    def __init__(self, tasks):
        if tasks is None:
            raise RuntimeError("Invalid tasks sequence - None")
        if len(tasks) == 0:
            raise RuntimeError("Invalid tasks sequence - empty list")

        self.tasks = tasks
        self.cost = 0

    def crossover(self, other):
        return self.__davisCrossover(other), other.__davisCrossover(self)

    def __davisCrossover(self, other):
        p1 = random.randint(0, len(self.tasks)-1)
        p2 = random.randint(0, len(self.tasks)-1)
        if p1 > p2:
            p1, p2 = p2, p1

        child = []
        for i in range(0, len(self.tasks)):
            child.append(Task(-1, -1, -1))

        for i in range(p1, p2):
            child[i] = self.tasks[i]

        originalHistogram = sequenceHistogram(self.tasks)

        def processSecondParentElementsForRange(start, stop):
            for i in range(start, stop):
                currentChildHistogram = sequenceHistogram(child)
                for j in range(0, len(other.tasks)):
                    index = i + j
                    if index >= len(other.tasks):
                        index -= len(other.tasks)
                    t = other.tasks[index]
                    if t.taskNumber() in currentChildHistogram and originalHistogram[t.taskNumber()] == currentChildHistogram[t.taskNumber()]:
                        continue
                    else:
                        child[i] = t
                        break

        processSecondParentElementsForRange(p2, len(child))
        processSecondParentElementsForRange(0, p1)
        return Genome(child)

    def mutate(self, sequence):
        if random.random() < MUTATION_PROBABILITY:
            self.validate(sequence)
            self.tasks = copy.deepcopy(self.tasks)
            pos1 = random.randint(0, len(self.tasks)-1)
            pos2 = random.randint(0, len(self.tasks)-1)
            self.tasks[pos1], self.tasks[pos2] = self.tasks[pos2], self.tasks[pos1]
            self.validate(sequence)

    def reset(self):
        self.cost = 0

    def size(self):
        return len(self.tasks)

    def validate(self, sequence):
        tmp = copy.deepcopy(sequence)
        if self.size() != len(sequence):
            raise Exception("Broken genome! Size: {}, expected size: {}".format(self.size(), len(sequence)))
        for task in self.tasks:
            index = -1
            for i in range(0, len(tmp)):
                if task.taskNumber() == tmp[i].taskNumber():
                    index = i
                    break
            if index == -1:
                raise Exception("Broken genome! Missing item: {}".format(task.taskNumber()))
            else:
                tmp.pop(index)

    def __lt__(self, other):
        return self.cost < other.cost


class GeneticAlgorithmTraverser(TraverserBase):
    def __init__(self, system):
        super().__init__(system)
        self._genes = []
        self.__currentGene = -1
        self.__genePoolSize = DEFAULT_POOL_SIZE

    def assignSequence(self, sequence):
        super().assignSequence(sequence)
        self.__genePoolSize = self.__calculatePoolSize(len(sequence))
        self.__generateGenes()
        self.__currentGene = 0

    def __calculatePoolSize(self, sequenceLength):
        return DEFAULT_POOL_SIZE

    def __generateGenes(self):
        g = Genome(self._currentSequence)
        self._genes.append(g)
        g.validate(self._initialSequence)
        for i in range(0, self.__genePoolSize-1):
            g = Genome(random.sample(self._currentSequence, k=len(self._currentSequence)))
            g.validate(self._initialSequence)
            self._genes.append(g)

    def nextIteration(self):
        self._genes[self.__currentGene].cost = self._currentCost
        if self._bestCost == -1 or self._bestCost > self._currentCost:
            self._acceptCurrentSolution()

        self.__currentGene += 1
        if self.__currentGene >= len(self._genes):
            l1 = len(self._genes)
            self.__depopulate()
            self.__mutate()
            self.__crossover()
            l2 = len(self._genes)
            if l1 != l2:
                raise Exception("Genes pool broken")
            self.__currentGene = 0

        self._currentSequence = self._genes[self.__currentGene].tasks
        self._tmpSequence = copy.deepcopy(self._currentSequence)
        self._currentCost = 0
        self._currentStatistics = TraverserStatistics(0, 0, 0, 0)

    def feedback(self, cost, collisions, timeInQueue, timeInPenalty, timeInTransition):
        super().feedback(cost, collisions, timeInQueue, timeInPenalty, timeInTransition)

    def __crossover(self):
        self.__validatePool()
        newGenes = []
        while len(self._genes) > 0:
            genom1 = self._genes.pop(random.randint(0, len(self._genes)-1))
            genom2 = self._genes.pop(random.randint(0, len(self._genes)-1))
            newGenom1, newGenom2 = genom1.crossover(genom2)
            genom1.reset()
            genom2.reset()
            newGenom1.validate(self._initialSequence)
            newGenom2.validate(self._initialSequence)
            genom1.validate(self._initialSequence)
            genom2.validate(self._initialSequence)
            newGenes.append(newGenom1)
            newGenes.append(newGenom2)
            newGenes.append(genom1)
            newGenes.append(genom2)
        self._genes = newGenes

    def __mutate(self):
        for genome in self._genes:
            genome.mutate(self._initialSequence)

    def __depopulate(self):
        self.__validatePool()
        self._genes.sort()
        self._genes = self._genes[0: int(len(self._genes) / 2)]
        self.__validatePool()

    def __validatePool(self):
        l = len(self._genes)
        if l % 2 != 0:
            raise RuntimeError("Invalid population size")
