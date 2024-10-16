from copy import deepcopy


class FakeOpcServer:
    def __init__(self, signalsObserver):
        self.__signalsDict = dict()
        self.__connections = list()
        self.__signalsObserver = signalsObserver

    def getSignalValue(self, signal):
        self.__signalsObserver.onSignalPolled(signal)
        return deepcopy(self.__signalsDict[signal])

    def setSignalValue(self, signal, value):
        self.__signalsObserver.onSignalChanged(signal)
        self.__signalsDict[signal] = value

    def onClientConnection(self, connectionString):
        self.__connections.append(connectionString)

    def connections(self):
        return self.__connections
