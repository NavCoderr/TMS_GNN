class AbstractOpcClientFactory:
    def __init__(self):
        pass

    def createOpcClient(self, errorSink):
        raise Exception("Not implemented!")