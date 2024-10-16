import threading, time, copy
from dataclasses import dataclass
from simulation.core.composition_root import CompositionRoot as SimulationRoot, SimulationInitInfo
from simulation.simpy_adapter.composition_root import CompositionRoot as SimpyRoot
from mes_adapter.composition_root import CompositionRoot as MesRoot, MesCompositionRootInitInfo
from agv_adapter.composition_root import CompositionRoot as AgvRoot
from tms.topology_builder import TopologyBuilder
from tcp_utils.client_utils import TcpClient
from storage.graph_storage import GraphStorage
from storage.mes_mapping_storage import MesMappingStorage
from storage.filesystem import Filesystem


class QueueObserver:
    def probeQueueState(self, queue, executorsViews, timePoint):
        pass


@dataclass
class TmsInitInfo:
    topologyDescriptionPath: str
    simulationMesIp: str
    simulationMesPort: int
    mesIp: str
    mesPort: int
    mesTasksMappingPath: str
    agvControllerIp: str
    agvControllerPort: int
    queueObserver: QueueObserver


class QueueObservingThread:
    def __init__(self, queueView, executorsManager, queueObserver):
        self.__working = False
        self.__thread = None
        self.__queueView = queueView
        self.__queueObserver = queueObserver
        self.__executorsManager = executorsManager

    def start(self):
        self.__working = True
        self.__thread = threading.Thread(target=self.__observeQueue)
        self.__thread.daemon = True
        self.__thread.start()

    def shutdown(self):
        self.__working = False
        self.__thread.join()
        self.__thread = None

    def __observeQueue(self):
        timePoint = 0
        interval = 1
        while self.__working:
            self.__queueObserver.probeQueueState(self.__queueView, self.__executorsManager.executorsViews(), timePoint)
            time.sleep(interval)
            timePoint += interval


class CompositionRoot:
    def __init__(self, networkSenderFactory=TcpClient):
        self.__simulationRoot = SimulationRoot()
        self.__simpyRoot = SimpyRoot(1000000)
        self.__mesRoot = MesRoot()
        self.__agvRoot = AgvRoot()
        self.__networkSenderFactory = networkSenderFactory
        self.__queueObservingThread = None
        self.__fs = Filesystem()
        self.__running = False

    def registerFile(self, file):
        self.__fs.addFile(file.name, file)

    def initialize(self, tmsInitInfo: TmsInitInfo):
        graphStorage = GraphStorage(self.__fs)
        mesMappingStorage = MesMappingStorage(self.__fs)
        graphStorage.read(tmsInitInfo.topologyDescriptionPath)
        mesMappingStorage.read(tmsInitInfo.mesTasksMappingPath)
        topologyBuilder = TopologyBuilder(self.__simpyRoot.simulation.env, graphStorage)
        self.__agvRoot.initialize(tmsInitInfo.agvControllerIp, tmsInitInfo.agvControllerPort)

        dependencies = {
            'agentsFactory': self.__simpyRoot.simpyAgentsFactory,
            'simulation': self.__simpyRoot.simulation,
            'taskExecutorsManager': self.__agvRoot.executorsManager()
        }
        simulationInitInfo = SimulationInitInfo(traverserName='geneticAlgorithm')
        self.__simulationRoot.initialize(dependencies, topologyBuilder, simulationInitInfo)
        mesInitInfo = MesCompositionRootInitInfo(dependencies={
            'mesDataSource': self.__networkSenderFactory(host=tmsInitInfo.mesIp, port=tmsInitInfo.mesPort),
            'simulationDataSource': self.__networkSenderFactory(host=tmsInitInfo.simulationMesIp, port=tmsInitInfo.simulationMesPort),
            'tasksQueue': self.__simulationRoot.tasksQueue(),
            'configuration': mesMappingStorage
        })
        self.__mesRoot.initialize(mesInitInfo)
        self.__queueObservingThread = QueueObservingThread(self.__simulationRoot.tasksQueue().queueView(), self.__simulationRoot.executorsManager(), tmsInitInfo.queueObserver)

    def start(self):
        self.__running = True
        self.__mesRoot.start()
        self.__queueObservingThread.start()
        self.__simulationRoot.start()

    def shutdown(self):
        self.__agvRoot.shutdown()
        self.__mesRoot.shutdown()
        self.__simulationRoot.shutdown()
        self.__queueObservingThread.shutdown()
        self.__running = False

    def isMesConnected(self):
        return self.__mesRoot.isMesConnected()

    def isSimulationMesConnected(self):
        return self.__mesRoot.isSimulationMesConnected()

    def isAgvHubConnected(self):
        return self.__agvRoot.isConnected()

    def isRunning(self):
        return self.__running