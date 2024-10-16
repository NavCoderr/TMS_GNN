from dataclasses import dataclass
from mes_adapter.mes_client import MesClient
from mes_adapter.tasks_source import MesTasksSource
from mes_adapter.requestToTaskMapper import RequestToTaskMapper
from mes_adapter.frame_parser import MesFrameParser
from mes_adapter.json_request_parser import JsonRequestParser


@dataclass
class MesCompositionRootInitInfo:
    dependencies: dict


class CompositionRoot:
    def __init__(self):
        self.__simulationClient = None
        self.__mesClient = None
        self.__tasksSource = None
        self.__requestMapper = None

    def initialize(self, initInfo: MesCompositionRootInitInfo):
        self.__requestMapper = RequestToTaskMapper(initInfo.dependencies['configuration'])
        self.__tasksSource = MesTasksSource(self.__requestMapper)
        self.__mesClient = MesClient(initInfo.dependencies['mesDataSource'], self.__tasksSource, MesFrameParser())
        self.__simulationClient = MesClient(initInfo.dependencies['simulationDataSource'], self.__tasksSource, JsonRequestParser())
        self.__tasksSource.setTasksQueue(initInfo.dependencies['tasksQueue'])

    def start(self):
        self.__mesClient.start()
        self.__simulationClient.start()

    def shutdown(self):
        self.__mesClient.kill()
        self.__simulationClient.kill()

    def isMesConnected(self):
        return self.__mesClient.isConnected()

    def isSimulationMesConnected(self):
        return self.__simulationClient.isConnected()