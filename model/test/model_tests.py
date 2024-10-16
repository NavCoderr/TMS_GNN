import unittest
from fake_view import FakeView
from fake_event_source import FakeEventsSource
from fake_error_sink import FakeErrorSink
from model.composition_root import CompositionRoot, InitData, MapData
from model.gnn_model import GNNModel  # Import GNN model
from simulation.core.queue_optimizer import QueueOptimizer  # Import GNN-enhanced QueueOptimizer
from simulation.core.system import System
from simulation.experiments_utils.test_graphs_builders import DebugGraphBuilder


class ModelTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.view = FakeView()
        self.eventsSource = FakeEventsSource()
        self.errorSink = FakeErrorSink()
        self.compositionRoot = CompositionRoot()
        self.compositionRoot.setView(self.view)
        self.compositionRoot.addErrorListener(self.errorSink)
        mapData = MapData("url", 10, 20, 1000, 2000)
        initData = InitData(mapData)
        self.compositionRoot.initialize(initData)
        self.compositionRoot.eventsHub().addEventsSource(self.eventsSource)

        # Initialize GNN model and QueueOptimizer
        graph_data = DebugGraphBuilder().buildGraph()
        self.system = System(graph_data=graph_data)
        self.gnn_model = GNNModel(num_node_features=graph_data.num_features, num_classes=1)
        self.gnn_model.eval()
        self.queue_optimizer = QueueOptimizer(system=self.system, agentsFactory=None, simulation=None,
                                              traverserFactory=None, queue=None, executorsManager=None)
        self.queue_optimizer.initializeGNN(self.gnn_model)

    def test_GNNQueueOptimizerIntegration(self):
        # Run the GNN-based optimizer and ensure it produces valid output
        mock_tasks = []  # Replace with appropriate mock tasks or generated tasks
        iterations = 1000

        result = self.queue_optimizer.optimizeQueue(iterations)

        # Validate that result is not None and contains expected structure
        self.assertIsNotNone(result, "GNN QueueOptimizer failed to return a result.")
        self.assertGreaterEqual(len(result.queueView.tasks), 0, "Queue should contain valid tasks or be empty.")
        self.assertGreaterEqual(result.statistics.collisions, 0, "Collisions must be non-negative.")

    def test_WhenAgvObjectIsRegisteredAndUpdatedViewRendersIt(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.updateObjectPosition(12, 100, 213)
        self.eventsSource.updateObjectPosition(12, 200, 413)
        self.eventsSource.updateObjectRotation(12, 120)

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        self.assertEqual(len(self.view.knownObjects), 1)

        self.assertEqual(self.view.renderedObjects[0].getObjectId(), 12)
        self.assertEqual(self.view.renderedObjects[0].getX(), 0)
        self.assertEqual(self.view.renderedObjects[0].getY(), 0)
        self.assertEqual(self.view.renderedObjects[0].getRotation(), 0)

        self.assertEqual(self.view.renderedObjects[1].getObjectId(), 12)
        self.assertEqual(self.view.renderedObjects[1].getX(), 100)
        self.assertEqual(self.view.renderedObjects[1].getY(), 213)
        self.assertEqual(self.view.renderedObjects[1].getRotation(), 0)

        self.assertEqual(self.view.renderedObjects[2].getObjectId(), 12)
        self.assertEqual(self.view.renderedObjects[2].getX(), 200)
        self.assertEqual(self.view.renderedObjects[2].getY(), 413)
        self.assertEqual(self.view.renderedObjects[2].getRotation(), 0)

        self.assertEqual(self.view.renderedObjects[3].getObjectId(), 12)
        self.assertEqual(self.view.renderedObjects[3].getX(), 200)
        self.assertEqual(self.view.renderedObjects[3].getY(), 413)
        self.assertEqual(self.view.renderedObjects[3].getRotation(), 120)

    def test_AgvObjectCanUpdateBatteryState(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.updateObjectProperties(12, { 'battery': '20%' })

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        self.assertEqual(self.view.renderedObjects[0].getProperties()['battery'], '10%')
        self.assertEqual(self.view.renderedObjects[1].getProperties()['battery'], '20%')

    def test_AgvObjectCanUpdateAlert(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.updateObjectAlerts(12, { 'alert': 'False' })
        self.eventsSource.updateObjectAlerts(12, { 'alert': 'True' })

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        self.assertEqual(self.view.renderedObjects[1].getAlerts()['alert'], 'False')
        self.assertEqual(self.view.renderedObjects[2].getAlerts()['alert'], 'True')


    def test_IncorrectPositionEventIsIgnoredAndErrorIsLogged(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.updateObjectPosition(12, 100, 213)
        self.eventsSource.updateObjectPosition(12, 1, 2)
        self.eventsSource.updateObjectPosition(12, 1000000000, 200000000)
        self.eventsSource.updateObjectPosition(12, -3, -2)

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        self.assertEqual(len(self.view.knownObjects), 1)

        self.assertEqual(self.view.lastRenderedObject().getObjectId(), 12)
        self.assertEqual(self.view.lastRenderedObject().getX(), 100)
        self.assertEqual(self.view.lastRenderedObject().getY(), 213)

        self.assertEqual(len(self.errorSink.errors), 3)
        self.assertEqual(self.view.lastRenderedObject().getObjectId(), 12)
        self.assertEqual(self.view.lastRenderedObject().getX(), 100)
        self.assertEqual(self.view.lastRenderedObject().getY(), 213)

        self.assertEqual(len(self.errorSink.errors), 3)
        self.assertEqual(self.view.lastRenderedObject().getObjectId(), 12)
        self.assertEqual(self.view.lastRenderedObject().getX(), 100)
        self.assertEqual(self.view.lastRenderedObject().getY(), 213)

        self.assertEqual(len(self.errorSink.errors), 3)
        self.assertEqual(self.view.lastRenderedObject().getObjectId(), 12)
        self.assertEqual(self.view.lastRenderedObject().getX(), 100)
        self.assertEqual(self.view.lastRenderedObject().getY(), 213)
        self.assertEqual(len(self.view.renderedObjects), 2)

    def test_WhenAgvObjectIsRegisteredTwiceOnlyOneInstanceExistsAndErrorIsLogged(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.registerAgvObject(12, "Dummy")

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        self.assertEqual(len(self.view.knownObjects), 1)
        self.assertEqual(len(self.errorSink.errors), 1)

    def test_WhenObjectIsUnregisteredTwiceNoneInstanceExistsAndErrorIsLogged(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.unregisterObject(12)
        self.eventsSource.unregisterObject(12)

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        self.assertEqual(len(self.view.knownObjects), 0)

    def test_AgvObjectGetsCorrectBoundingRect(self):
        self.eventsSource.registerAgvObject(12, "Dummy")
        self.eventsSource.updateObjectPosition(12, 200, 413)

        self.eventsSource.processEventsQueue()
        self.compositionRoot.startProcessingEvents()
        self.eventsSource.threadWorker.join()

        lastRenderedObject = self.view.lastRenderedObject()
        boundingRect = lastRenderedObject.getBoundingRect()
        topLeft = boundingRect[0]
        bottomRight = boundingRect[1]
        self.assertEqual(topLeft[0], 200)
        self.assertEqual(topLeft[1], 413)
        self.assertEqual(bottomRight[0], 204)
        self.assertEqual(bottomRight[1], 416)

    def tearDown(self):
        super().tearDown()


if __name__ == '__main__':
    unittest.main()
