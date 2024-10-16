from opc_adapter.opc_client_factory import AbstractOpcClientFactory


class FakeOpcClient:
    def __init__(self, fakeOpcServer):
        self.__fakeOpcServer = fakeOpcServer

    def connect(self, connectionString):
        self.__fakeOpcServer.onClientConnection(connectionString)

    def getSignalValue(self, signal):
        return self.__fakeOpcServer.getSignalValue(signal)


class FakeOpcClientsFactory(AbstractOpcClientFactory):
    
    def __init__(self, fakeOpcServer):
        super().__init__()
        self.__fakeOpcServer = fakeOpcServer

    def createOpcClient(self, errorSink):
        return FakeOpcClient(self.__fakeOpcServer)
