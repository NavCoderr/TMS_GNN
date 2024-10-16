from mes_adapter.test_utils.test_data import getTestFrame


class FakeTcpClientsManager:
    def __init__(self):
        self.hosts = dict()

    def __call__(self, *args, **kwargs):
        try:
            host = args[0]
            port = args[1]
        except IndexError:
            host = kwargs['host']
            port = kwargs['port']
        self.hosts[host] = FakeTcpClient(host, port)
        return self.hosts[host]


class FakeTcpClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sentData = list()
        self.__packetsToRead = 0
        self.__readMode = 'mes'

    def setPacketsToRead(self, number):
        self.__packetsToRead = number

    def setReadMode(self, mode):
        self.__readMode = mode

    def readDataFromServer(self):
        if self.__readMode == 'mes':
            if self.__packetsToRead > 0:
                self.__packetsToRead -= 1
                return getTestFrame()
        elif self.__readMode == 'agv':
            if self.__packetsToRead > 0:
                self.__packetsToRead -= 1
                return int(0).to_bytes(1, 'big')
            return int(1).to_bytes(1, 'big')
        return None

    def sendDataToServer(self, data):
        self.sentData.append(data)