from dataclasses import dataclass
import json


@dataclass
class OrderDefinition:
    orderId: int
    source: int
    destination: int


class MesMappingStorage:
    def __init__(self, filesystem):
        self.__data = None
        self.__filename = ""
        self.__fs = filesystem

    def read(self, filename):
        self.__filename = filename
        self.__data = json.loads(self.__fs.readFile(self.__filename))

    def ordersDefintions(self):
        nodes = list()
        for node in self.__data['orders']:
            nodes.append(OrderDefinition(int(node['orderId']), int(node['source']), int(node['destination'])))
        return nodes
