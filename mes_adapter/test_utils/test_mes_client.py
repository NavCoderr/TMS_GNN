from tcp_utils.client_utils import TcpClient
from mes_adapter.composition_root import CompositionRoot, MesCompositionRootInitInfo


class LoggingQueue:
    def enqueue(self, task):
        print("Executing task: {}".format(task))


class TestMesClient:
    def __init__(self):
        self.__logger = LoggingQueue()
        self.__root = CompositionRoot()

    def start(self, mesIp, mesPort):
        mesInitInfo = MesCompositionRootInitInfo(tasksMapperConfigPath='', dependencies={
            'mesDataSource': TcpClient(host=mesIp, port=mesPort),
            'tasksQueue': self.__logger
        })

        self.__root.initialize(mesInitInfo)
        self.__root.start()
        while True:
            pass


client = TestMesClient()
client.start('localhost', 1234)