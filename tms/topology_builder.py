from simulation.simpy_adapter.node import Node
from simulation.core.system_builder import *


class TopologyBuilder:

    def __init__(self, env, dataSource):
        self.__env = env
        self.__ds = dataSource

    def build(self, systemBuilder):

        nodesIndices = []
        nodesDescriptions = self.__ds.nodesDescriptions()
        for nodeDescription in nodesDescriptions:
            nodesIndices.append(nodeDescription.index)
            systemBuilder.addVertex(Vertex(node=Node(env=self.__env, serviceTime=nodeDescription.serviceTime, index=nodeDescription.index), name='unused'))

        maxIndex = max(nodesIndices)
        for i in range(0, maxIndex):
            if i not in nodesIndices:
                systemBuilder.addVertex(Vertex(node=Node(env=self.__env, serviceTime=0, index=i), name='gapFillingNode'))

        for edgeDescription in self.__ds.edgesDescriptions():
            systemBuilder.addEdge(Edge(source=edgeDescription.firstNode, target=edgeDescription.secondNode, weight=edgeDescription.transitionTime, name='unused'))

