import opcua
from agv_type_definitions import *
from agv_opc import AgvOpc


class OpcServer:
    def __init__(self):
        self.__agvType = None
        self.__server = opcua.Server()
        self.__index = self.__server.register_namespace("opc.tcp://127.0.0.1:4840/fake_agv/")
        self.__createAgvType()
        self.__agvsFolder = self.__server.get_objects_node().add_folder(self.__index, "AGVs")
        self.__objects = dict()

    def addAgvObject(self, name):
        self.__objects[name] = self.__agvsFolder.add_object(self.__index, name, self.__agvType)

    def __createAgvType(self):
        self.__agvType = self.__server.nodes.base_object_type.add_object_type(self.__index, "AGV")
        self.__agvType.add_variable(self.__index, x_signal, 1.0).set_modelling_rule(True)
        self.__agvType.add_variable(self.__index, y_signal, 1.5).set_modelling_rule(True)
        self.__agvType.add_variable(self.__index, heading_signal, 1.0).set_modelling_rule(True)
        self.__agvType.add_property(self.__index, source_node_signal, "A").set_modelling_rule(True)
        self.__agvType.add_property(self.__index, destination_node_signal, "B").set_modelling_rule(True)
        self.__agvType.add_property(self.__index, battery_signal, "10%").set_modelling_rule(True)
        self.__agvType.add_property(self.__index, state_signal, "Idle").set_modelling_rule(True)

    def start(self):
        self.__server.start()

    def getAgvOpc(self, objectName):
        return AgvOpc(self.__objects[objectName], self.__index)


server = OpcServer()
server.start()

