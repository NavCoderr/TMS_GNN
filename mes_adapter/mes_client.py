import threading, time
from mes_adapter.request_parser import *
from mes_adapter.frame_parser import MesFrameParser
from mes_adapter.tasks_source import MesTasksSource


class MesClient:

    def __init__(self, mesDataSource, tasksSource: MesTasksSource, dataParser: RequestParser):
        self.__mesDataSource = mesDataSource
        self.__pollingThread = None
        self.__running = True
        self.__requestParser = dataParser
        self.__tasksSource = tasksSource
        self.__createThread()

    def __createThread(self):
        self.__pollingThread = threading.Thread(target=self.__pollMes)
        self.__pollingThread.daemon = True

    def __pollMes(self):
        while self.__running:
            try:
                while not self.isConnected():
                    self.__mesDataSource.connect()
                    time.sleep(5)
                    if not self.__running: # TMS killed in the meantime
                        break
                self.__mesDataSource.sendDataToServer(bytes("TMS_READY", encoding='ASCII'))
                data = self.__mesDataSource.readDataFromServer()
                if data is not None:
                    parsed = self.__requestParser.parse(data)
                    productionOrderId = parsed.orderId
                    taskId = parsed.uniqueId
                    self.__tasksSource.handleRequest(productionOrderId, taskId)
                    confirmation = bytes(str(productionOrderId), encoding='ASCII')
                    self.__mesDataSource.sendDataToServer(confirmation)
                time.sleep(1)
            except ConnectionRefusedError as e:
                pass

    def start(self):
        self.__pollingThread.start()

    def kill(self):
        self.__running = False
        self.__pollingThread.join()

    def isConnected(self):
        return self.__mesDataSource.isConnected()