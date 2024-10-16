from simulation.simpy_adapter.node import Node
from simulation.core.system_builder import *
import random


class GraphNodesWeightsManager:
    def __init__(self, nodesNumber, min, max):
        self.__nodesNumber = nodesNumber
        self.__weights = None
        self.__min = min
        self.__max = max

    def getWeight(self, index):
#        return random.uniform(self.__min, self.__max)
        if self.__weights is None:
            self.__generateWeights()
        return self.__weights[index]

    def __generateWeights(self):
        self.__weights = dict()
        for i in range(0, self.__nodesNumber):
            self.__weights[i] = random.uniform(self.__min, self.__max)


veryLongServiceTimeWeightsManager = GraphNodesWeightsManager(1000, 100, 5000)
longServiceTimeWeightsManager = GraphNodesWeightsManager(1000, 1, 10)
shortServiceTimeWeightsManager = GraphNodesWeightsManager(1000, 1, 10)


class DebugGraphBuilder:

    def __init__(self, nodesNumber):
        self.__nodesNumber = nodesNumber
        self.__env = None

    def setEnvironment(self, env):
        self.__env = env
        return self

    def build(self, systemBuilder):
        n = self.__nodesNumber
        for i in range(0, n):
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self.__env, serviceTime=i, index=i)))

        for i in range(0, n):
            for j in range(0, n):
                systemBuilder.addEdge(Edge(name="unused", source=i, target=j, weight=i*10 + j))


class VeryLongServiceTimeFullGraphBuilder:

    def __init__(self, nodesNumber):
        self.__nodesNumber = nodesNumber
        self.__env = None

    def setEnvironment(self, env):
        self.__env = env
        return self

    def build(self, systemBuilder):
        n = self.__nodesNumber
        for i in range(0, n):
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self.__env, serviceTime=veryLongServiceTimeWeightsManager.getWeight(i), index=i)))

        for i in range(0, n):
            for j in range(0, n):
                systemBuilder.addEdge(Edge(name="unused", source=i, target=j, weight=random.uniform(10, 100)))


class LongServiceTimeFullGraphBuilder:

    def __init__(self, nodesNumber):
        self.__nodesNumber = nodesNumber
        self.__env = None

    def setEnvironment(self, env):
        self.__env = env
        return self

    def build(self, systemBuilder):
        n = self.__nodesNumber
        for i in range(0, n):
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self.__env, serviceTime=longServiceTimeWeightsManager.getWeight(i), index=i)))

        for i in range(0, n ):
            for j in range(0, n ):
                systemBuilder.addEdge(Edge(name="unused", source=i, target=j, weight=random.uniform(10, 100)))


class ShortServiceTimeFullGraphBuilder:

    def __init__(self, nodesNumber):
        self.__nodesNumber = nodesNumber
        self.__env = None

    def setEnvironment(self, env):
        self.__env = env
        return self

    def build(self, systemBuilder):
        n = self.__nodesNumber
        for i in range(0, n):
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self.__env, serviceTime=shortServiceTimeWeightsManager.getWeight(i), index=i)))

        for i in range(0, n ):
            for j in range(0, n ):
                systemBuilder.addEdge(Edge(name="unused", source=i, target=j, weight=random.uniform(10, 100)))


class GraphWithIntermediatePointsBuilderBase:
    def __init__(self, nodesNumber):
        self._nodesNumber = nodesNumber
        self._env = None
        self._globalIndex = 0

    def setEnvironment(self, env):
        self._env = env
        return self

    def buildConnectionBetweenStations(self, systemBuilder, source, target, branches, branchLengthGenerator):
        def intermediateEdgesWeight():
            return 1
        for _ in range(0, branches):
            self.buildSingleBranchBetweenStations(systemBuilder, source, target, branchLengthGenerator(), intermediateEdgesWeight)

    def buildSingleBranchBetweenStations(self, systemBuilder, source, target, branchLength, intermediateEdgesWeightGenerator):
        i = 0
        currentIndex = source
        while i < branchLength:
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self._env, serviceTime=0, index=self._globalIndex)))
            systemBuilder.addEdge(Edge(name="unused", source=currentIndex, target=self._globalIndex, weight=intermediateEdgesWeightGenerator()))
            currentIndex = self._globalIndex
            self._globalIndex += 1
            i += 1
        systemBuilder.addEdge(Edge(name="unused", source=currentIndex, target=target, weight=random.uniform(10, 100)))


class ShortServiceTimeFullGraphWithBranchesBuilder(GraphWithIntermediatePointsBuilderBase):
    branches = 3

    def build(self, systemBuilder):
        self._globalIndex = 0

        def branchLengthGenerator():
            return random.randint(3, 5)

        n = self._nodesNumber
        indices = []
        for i in range(0, n):
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self._env, serviceTime=shortServiceTimeWeightsManager.getWeight(i), index=self._globalIndex)))
            indices.append(self._globalIndex)
            self._globalIndex += 1

        connections = []
        for i in indices:
            for j in indices:
                if j > i:
                    connections.append((i, j))

        for connection in connections:
            self.buildConnectionBetweenStations(systemBuilder, connection[0], connection[1], ShortServiceTimeFullGraphWithBranchesBuilder.branches, branchLengthGenerator)



class TreeGraphBuilder:

    def __init__(self, switches, hostsPerSwitch):
        self.__switchesNumber = switches
        self.__hostsNumberPerSwitch = hostsPerSwitch
        self.__env = None
        self.__switches = list()
        self.__hosts = list()

    def setEnvironment(self, env):
        self.__env = env
        return self

    def build(self, systemBuilder):
        index = 0
        n = self.__switchesNumber
        for i in range(0, n):
            self.__switches.append(index)
            systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self.__env, serviceTime=10, index=index)))
            index += 1

        for i in range(0, n ):
            for j in range(0, n ):
                systemBuilder.addEdge(Edge(name="unused", source=i, target=j, weight=10))

        n = self.__hostsNumberPerSwitch
        for switch in self.__switches:
            for i in range(0, n):
                systemBuilder.addVertex(Vertex(name="unused", node=Node(env=self.__env, serviceTime=0, index=index)))
                self.__hosts.append(index)
                systemBuilder.addEdge(Edge(name="unused", source=index, target=switch, weight=1))
                index += 1
