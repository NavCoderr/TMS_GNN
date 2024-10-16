from fake_agv.agv_logic import AgvLogic
from fake_agv.agv_logic_2_opc_adapter import AgvLogic2OpcAdapter


class AgvBuilder:
    def __init__(self, opcServer, agvController, nodes, adapters):
        self.__server = opcServer
        self.__controller = agvController
        self.__nodes = nodes
        self.__adapters = adapters

    def addAgv(self, name):
        agvLogic = AgvLogic(self.__nodes)
        self.__controller.addAgv(name, agvLogic)
        self.__server.addAgvObject(name)
        self.__adapters[name] = AgvLogic2OpcAdapter(agvLogic=agvLogic, agvOpc=self.__server.getAgvOpc(name))