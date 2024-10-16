from fake_agv.opc_server import OpcServer
from fake_agv.agv_controller import AgvController
from fake_agv.agv_builder import AgvBuilder


class CompositionRoot:
    def __init__(self):
        self.__opcServer = OpcServer()
        self.__agvController = AgvController()
        self.__opcAdapters = dict()
        self.__nodes = list()

    def getBuilder(self):
        return AgvBuilder(opcServer=self.__opcServer, agvController=self.__agvController, nodes=self.__nodes, adapters=self.__opcAdapters)