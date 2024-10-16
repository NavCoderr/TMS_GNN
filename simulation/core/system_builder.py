from simulation.core.system import System
from collections import namedtuple


Vertex = namedtuple('Vertex', 'name node')
Edge = namedtuple('Edge', 'name source target weight')


class SystemBuilder:
    def __init__(self):
        self.__system = System()
        self.__system.graph.es["weight"] = 1.0

    def addVertex(self, vertex):
        self.__system.graph.add_vertex(node=vertex.node)

    def addEdge(self, edge):
        self.__system.graph.add_edge(edge.source, edge.target, agents={}, executors={})
        self.__system.graph[edge.source, edge.target] = edge.weight

    def system(self):
        return self.__system

    