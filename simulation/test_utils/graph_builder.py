from simulation.simpy_adapter.node import Node
from simulation.core.system_builder import *


class GraphBuilder:

    def __init__(self, env):
        self.__env = env
        self.__nodesNumber = None
        self.__serviceTime = None
        self.__edgeWeight = None

    def setBuildParameters(self, nodesNumber, serviceTime, edgeWeight):
        self.__nodesNumber = nodesNumber
        self.__serviceTime = serviceTime
        self.__edgeWeight = edgeWeight

    def build(self, systemBuilder):
        n = self.__nodesNumber
        for i in range(0, n):
            systemBuilder.addVertex(Vertex(node=Node(env=self.__env, serviceTime=self.__serviceTime, index=i), name='unused'))

        for i in range(0, n ):
            for j in range(0, n ):
                systemBuilder.addEdge(Edge(source=i, target=j, weight=self.__edgeWeight, name='unused'))
