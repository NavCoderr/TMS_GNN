import unittest
from tms.test.fake_system import FakeSystem
from storage.graph_storage import GraphStorage
from tms.topology_builder import TopologyBuilder


class TasksSchedulingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.__fakeSystem = FakeSystem()
        self.__graphStorage = GraphStorage()
        self.__graphStorage.read('testGraph.json')

    def test_buildsSimpleGraph(self):

        def checkVertex(vertex, expectedIndex, expectedServiceTime):
            self.assertEqual(expectedIndex, vertex.node.index)
            self.assertEqual(expectedServiceTime, vertex.node.serviceTime)

        def checkEdge(edge, expectedSource, expectedTarget, expectedWeight):
            self.assertEqual(expectedSource, edge.source)
            self.assertEqual(expectedTarget, edge.target)
            self.assertEqual(expectedWeight, edge.weight)

        builderAtTest = TopologyBuilder(None, self.__graphStorage)
        builderAtTest.build(self.__fakeSystem)
        self.assertEqual(4, len(self.__fakeSystem.vertices))
        self.assertEqual(6, len(self.__fakeSystem.edges))

        checkVertex(self.__fakeSystem.vertices[0], 0, 10.0)
        checkVertex(self.__fakeSystem.vertices[1], 1, 5.0)
        checkVertex(self.__fakeSystem.vertices[2], 2, 20.0)
        checkVertex(self.__fakeSystem.vertices[3], 3, 10.0)

        checkEdge(self.__fakeSystem.edges[0], 0, 1, 10.0)
        checkEdge(self.__fakeSystem.edges[1], 0, 2, 15.0)
        checkEdge(self.__fakeSystem.edges[2], 0, 3, 5.0)
        checkEdge(self.__fakeSystem.edges[3], 1, 2, 20.0)
        checkEdge(self.__fakeSystem.edges[4], 1, 3, 25.0)
        checkEdge(self.__fakeSystem.edges[5], 2, 3, 20.0)