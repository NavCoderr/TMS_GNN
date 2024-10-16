import unittest, copy, time
from tms.composition_root import CompositionRoot as TmsRoot
from tms.composition_root import TmsInitInfo, QueueObserver
from tms.test.fake_tcp_client import FakeTcpClient, FakeTcpClientsManager

mesIp = '10.10.0.0'
initInfoPrototype = TmsInitInfo(topologyDescriptionPath='testGraph.json',
                                mesIp=mesIp,
                                mesPort=8181,
                                mesTasksMappingPath='unused',
                                agvConnectionsData=[
                                    ('10.0.0.1', 80),
                                    ('10.0.0.2', 80),
                                    ('10.0.0.3', 80)
                                ],
                                queueObserver=QueueObserver())


class TasksSchedulingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.fakeTcpClientsManager = FakeTcpClientsManager()
        self.__tmsRoot = TmsRoot(self.fakeTcpClientsManager)
        self.initInfo = copy.deepcopy(initInfoPrototype)

    def test_initializesCorrectly(self):
        self.__tmsRoot.initialize(self.initInfo)
        self.assertEqual(4, len(self.fakeTcpClientsManager.hosts))

    def test_whenMesSendsFrameTheTasksAreDistributedToAgv(self):
        self.__tmsRoot.initialize(self.initInfo)
        self.assertEqual(4, len(self.fakeTcpClientsManager.hosts))

        self.fakeTcpClientsManager.hosts[mesIp].setPacketsToRead(1)
        self.fakeTcpClientsManager.hosts['10.0.0.1'].setReadMode('agv')
        self.fakeTcpClientsManager.hosts['10.0.0.1'].setPacketsToRead(5)
        self.fakeTcpClientsManager.hosts['10.0.0.2'].setReadMode('agv')
        self.fakeTcpClientsManager.hosts['10.0.0.2'].setPacketsToRead(5)
        self.fakeTcpClientsManager.hosts['10.0.0.3'].setReadMode('agv')
        self.fakeTcpClientsManager.hosts['10.0.0.3'].setPacketsToRead(5)
        self.__tmsRoot.start()
        time.sleep(3)
        self.__tmsRoot.shutdown()

        handledHost = ''
        if len(self.fakeTcpClientsManager.hosts['10.0.0.1'].sentData) > 0:
            handledHost = '10.0.0.1'
        elif len(self.fakeTcpClientsManager.hosts['10.0.0.2'].sentData) > 0:
            handledHost = '10.0.0.2'
        elif len(self.fakeTcpClientsManager.hosts['10.0.0.3'].sentData) > 0:
            handledHost = '10.0.0.3'

        self.assertNotEqual(handledHost, '')
        self.assertEqual('0', str(int.from_bytes(self.fakeTcpClientsManager.hosts[handledHost].sentData[0], 'big')))
        self.assertEqual('1', str(int.from_bytes(self.fakeTcpClientsManager.hosts[handledHost].sentData[1], 'big')))

    def test_whenMesSendsMultipleFramesTheTasksAreDistributedToManyAgvs(self):

        def setupAgvHost(hostIp, packetsToRead):
            self.fakeTcpClientsManager.hosts[hostIp].setReadMode('agv')
            self.fakeTcpClientsManager.hosts[hostIp].setPacketsToRead(packetsToRead)

        self.__tmsRoot.initialize(self.initInfo)
        self.assertEqual(4, len(self.fakeTcpClientsManager.hosts))

        self.fakeTcpClientsManager.hosts[mesIp].setPacketsToRead(3)
        setupAgvHost('10.0.0.1', 10)
        setupAgvHost('10.0.0.2', 10)
        setupAgvHost('10.0.0.3', 10)
        self.__tmsRoot.start()
        time.sleep(5)
        self.__tmsRoot.shutdown()

        self.assertEqual(2, len(self.fakeTcpClientsManager.hosts['10.0.0.3'].sentData))
        self.assertEqual('0', self.fakeTcpClientsManager.hosts['10.0.0.3'].sentData[0])
        self.assertEqual('1', self.fakeTcpClientsManager.hosts['10.0.0.3'].sentData[1])
        self.assertEqual(2, len(self.fakeTcpClientsManager.hosts['10.0.0.1'].sentData))
        self.assertEqual('0', self.fakeTcpClientsManager.hosts['10.0.0.1'].sentData[0])
        self.assertEqual('1', self.fakeTcpClientsManager.hosts['10.0.0.1'].sentData[1])
        self.assertEqual(2, len(self.fakeTcpClientsManager.hosts['10.0.0.2'].sentData))
        self.assertEqual('0', self.fakeTcpClientsManager.hosts['10.0.0.2'].sentData[0])
        self.assertEqual('1', self.fakeTcpClientsManager.hosts['10.0.0.2'].sentData[1])

