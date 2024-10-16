from agv_type_definitions import *


class AgvOpc:
    def __init__(self, node, index):
        self.__node = node
        self.__index = index

    def setX(self, newValue):
        self.__setSignal(x_signal, newValue)

    def setY(self, newValue):
        self.__setSignal(y_signal, newValue)

    def __getProxy(self, signal):
        return self.__node.get_child("{}:{}".format(self.__index, signal))

    def __setSignal(self, signal, newValue):
        self.__getProxy(signal).set_value(newValue)