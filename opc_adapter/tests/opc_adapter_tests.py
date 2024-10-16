import unittest
import time

from opc_adapter.composition_root import CompositionRoot
from fake_opc_server import FakeOpcServer
from fake_client_factory import FakeOpcClientsFactory
from fake_events_hub import FakeEventsHub

X_SIGNAL = "signals_x_coordinate"
Y_SIGNAL = "signals_y_coordinate"


def objectCreationData():
    registerData = dict()
    registerData['width'] = 10
    registerData['height'] = 20
    registerData['type'] = "AGV"
    registerData['xSignal'] = X_SIGNAL
    registerData['ySignal'] = Y_SIGNAL
    registerData['connectionString'] = "opc.tcp://localhost:1234"
    registerData['updateInterval'] = 0.1
    return registerData


class SignalsPolledObserver:
    def __init__(self, observedSignals):
        self.__observedSignals = {}
        for signal in observedSignals:
            self.__observedSignals[signal] = False

    def onSignalPolled(self, signal):
        self.__observedSignals[signal] = True

    def onSignalChanged(self, signal):
        self.__observedSignals[signal] = False

    def allSignalsPolled(self):
        for signalName in self.__observedSignals:
            if not self.__observedSignals[signalName]:
                return False
        return True


class OpcAdapterTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.signalsObserver = SignalsPolledObserver([X_SIGNAL, Y_SIGNAL])
        self.server = FakeOpcServer(self.signalsObserver)
        self.factory = FakeOpcClientsFactory(self.server)
        self.eventsHub = FakeEventsHub()
        self.compositionRoot = CompositionRoot(self.eventsHub, self.factory)
        self.testObjects = list()

    def createTestObject(self, objectId, data):
        testObject = self.compositionRoot.objectsFactory().createObject(objectId=objectId, registerData=data, errorSink=None)
        self.testObjects.append(testObject)
        return testObject

    def test_WhenOpcObjectIsCreatedConnectionToServerIsOpened(self):
        self.createTestObject(1, objectCreationData())
        self.assertEqual(1, len(self.server.connections()))
        self.assertEqual("opc.tcp://localhost:1234", self.server.connections()[0])

    def test_WhenOpcObjectIsCreatedItReadsCurrentPositionFromServer(self):
        self.server.setSignalValue(X_SIGNAL, 10)
        self.server.setSignalValue(Y_SIGNAL, 20)
        self.assertEqual(self.server.getSignalValue(X_SIGNAL), 10)
        self.assertEqual(self.server.getSignalValue(Y_SIGNAL), 20)
        self.createTestObject(1, objectCreationData())
        self.assertEqual(1, len(self.eventsHub.events))

    def test_WhenObjectsPositionChangeInServerPositionChangeEventIsGenerated(self):
        self.server.setSignalValue(X_SIGNAL, 10)
        self.server.setSignalValue(Y_SIGNAL, 20)

        self.createTestObject(1, objectCreationData())

        self.assertEqual(self.eventsHub.lastEvent().x, 10)
        self.assertEqual(self.eventsHub.lastEvent().y, 20)

        self.server.setSignalValue(X_SIGNAL, 100)
        while not self.signalsObserver.allSignalsPolled():
            time.sleep(0.05)

        self.assertEqual(self.eventsHub.lastEvent().x, 100)
        self.assertEqual(self.eventsHub.lastEvent().y, 20)

        self.server.setSignalValue(Y_SIGNAL, 200)
        while not self.signalsObserver.allSignalsPolled():
            time.sleep(0.05)

        self.assertEqual(self.eventsHub.lastEvent().x, 100)
        self.assertEqual(self.eventsHub.lastEvent().y, 200)

        self.server.setSignalValue(X_SIGNAL, 1)
        self.server.setSignalValue(Y_SIGNAL, 2)
        while not self.signalsObserver.allSignalsPolled():
            time.sleep(0.05)

        self.assertEqual(self.eventsHub.lastEvent().x, 1)
        self.assertEqual(self.eventsHub.lastEvent().y, 2)

    def tearDown(self):
        for testObject in self.testObjects:
            testObject.shutdown()
        super().tearDown()


if __name__ == '__main__':
    unittest.main()
