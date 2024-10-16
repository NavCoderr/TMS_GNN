import sys, signal, csv, time
from tms.composition_root import CompositionRoot, TmsInitInfo, QueueObserver
from tms.test_utils.logger import Logger


class CliQueueObserver(QueueObserver):
    def __init__(self, logFilename):
        self.__qlens = list()
        self.__fields = ['time', 'qlen']
        self.__filename = logFilename

    def probeQueueState(self, queue, executorsViews, timePoint):
        global logger
        qlen = len(queue.tasksList()) + len(queue.pendingTasksList())
        self.__qlens.append({'time': round(timePoint, 2), 'qlen': qlen})
        if len(self.__qlens) > 100:
            self.save()

    def save(self):
        with open(self.__filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.__fields)
            writer.writerows(self.__qlens)
            self.__qlens.clear()


class SignalHandler:
    def __init__(self):
        self.s1 = signal.signal(signal.SIGINT, self.handleExit)
        self.s2 = signal.signal(signal.SIGTERM, self.handleExit)

        self.killed = False

    def handleExit(self, *args):
        self.killed = True


logger = Logger("tms_log.txt")
mesConnectionString = sys.argv[1].split(':')
mesIp, mesPort = mesConnectionString[0], int(mesConnectionString[1])
simulationMesConnectionString = sys.argv[2].split(':')
simulationMesIp, simulationMesPort = simulationMesConnectionString[0], int(simulationMesConnectionString[1])
agvControllerConnectionString = sys.argv[3].split(':')
agvControllerIp, agvControllerPort = agvControllerConnectionString[0], int(agvControllerConnectionString[1])

qlensObserver = CliQueueObserver('qlens.csv')
sigIntHandler = SignalHandler()
logger.logLine("Starting TMS with MES: {}:{}, simulation MES: {}:{} and AGV controller: {}:{}".format(
    mesIp,
    mesPort,
    simulationMesIp,
    simulationMesPort,
    agvControllerIp,
    agvControllerPort))
initInfo = TmsInitInfo(topologyDescriptionPath=sys.argv[4], mesIp=mesIp, mesPort=mesPort,
                       mesTasksMappingPath=sys.argv[5], agvControllerIp=agvControllerIp,
                       agvControllerPort=agvControllerPort, queueObserver=qlensObserver,
                       simulationMesIp=simulationMesIp, simulationMesPort=simulationMesPort)
tmsRoot = CompositionRoot()
tmsRoot.initialize(tmsInitInfo=initInfo)
tmsRoot.start()
while not sigIntHandler.killed:
    time.sleep(1)
    qlensObserver.save()

tmsRoot.shutdown()
logger.logLine("SIGINT or SIGTERM handled")
sys.exit(0)
