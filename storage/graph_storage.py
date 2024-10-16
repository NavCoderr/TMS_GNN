from dataclasses import dataclass
import json


@dataclass
class NodeInfo:
    index: int
    serviceTime: float


@dataclass
class EdgeInfo:
    firstNode: int
    secondNode: int
    transitionTime: float


class GraphStorage:
    def __init__(self, filesystem):
        self.__data = None
        self.__filename = ""
        self.__fs = filesystem

    def read(self, filename):
        self.__filename = filename
        self.__data = json.loads(self.__fs.readFile(self.__filename))

    def nodesDescriptions(self):
        nodes = list()
        for node in self.__data['nodes']:
            nodes.append(NodeInfo(int(node['index']), float(node['serviceTime'])))
        return nodes

    def edgesDescriptions(self):
        edges = list()
        for edge in self.__data['edges']:
            nodes = edge['nodes']
            if len(nodes) != 2:
                raise Exception("Invalid graph description!")
            edges.append(EdgeInfo(int(nodes[0]), int(nodes[1]), float(edge['transitionTime'])))
        return edges

