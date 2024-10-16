import unittest, time
from simulation.core.composition_root import CompositionRoot as CoreRoot
from simulation.simpy_adapter.composition_root import CompositionRoot as SimpyRoot
from simulation.test_utils.graph_builder import GraphBuilder
from simulation.core.tests.fake_tasks_executor import FakeTasksExecutor, fakeExecutors
from simulation.test_utils.tasks_generator import generateTasksQueue
from simulation.core.tests.fake_tasks_source import FakeTasksSource


class TasksSchedulingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.__coreRoot = CoreRoot()
        self.__simpyRoot = SimpyRoot(1000000)
        self.__graphBuilder = GraphBuilder(self.__simpyRoot.simulation.env)
        self.__coreRoot.start()

    def tearDown(self) -> None:
        fakeExecutors.clear()
        self.__coreRoot.shutdown()

    def initialize(self, executorsNumber, nodesNumber, serviceTime, transitTime):
        self.__graphBuilder.setBuildParameters(nodesNumber, serviceTime, transitTime)
        dependencies = {
            'agentsFactory': self.__simpyRoot.simpyAgentsFactory,
            'simulation': self.__simpyRoot.simulation,
            'tasksExecutorsFactory': FakeTasksExecutor
        }
        initInfo = {'executorsNumber': executorsNumber}
        self.__coreRoot.initialize(dependencies, self.__graphBuilder, initInfo)

    def test_initializationCreatesTasksExecutors(self):
        self.initialize(3, 10, 10, 1)
        self.assertEqual(len(fakeExecutors), 3)

    def test_newTasksAreDistributedToTasksExecutors(self):
        nodesNumber = 10
        self.initialize(3, nodesNumber, 10, 1)
        self.__coreRoot.tasksQueue().batchEnqueue(generateTasksQueue(12, nodesNumber))
        self.assertEqual(len(fakeExecutors), 3)
        self.__coreRoot.tasksScheduler().waitForQueueProcessed()
        for executorId in fakeExecutors:
            validResult = fakeExecutors[executorId].getExecutedTasksNumber() == 7 or fakeExecutors[executorId].getExecutedTasksNumber() == 8
            self.assertEqual(True, validResult)

    def test_distributesTasksToExecutorsInRealTime(self):
        nodesNumber = 10
        tasksSource = FakeTasksSource()
        tasks = generateTasksQueue(100, nodesNumber)
        for task in tasks:
            tasksSource.addTask(task)

        self.initialize(2, nodesNumber, 10, 1)
        self.__coreRoot.tasksScheduler().addTasksSource(tasksSource)
        self.__coreRoot.tasksQueue().batchEnqueue(generateTasksQueue(10, nodesNumber))
        self.assertEqual(len(fakeExecutors), 2)

        tasksSource.startProcessing(0)

        self.__coreRoot.tasksScheduler().waitForQueueProcessed()

        executedTasksCount = 0
        for executorId in fakeExecutors:
            executedTasksCount += fakeExecutors[executorId].getExecutedTasksNumber()

        executedCorrectNumberOfTasks = 200 < executedTasksCount < 220
        self.assertEqual(True, executedCorrectNumberOfTasks, "Executed tasks: {}".format(executedTasksCount))


if __name__ == '__main__':
    unittest.main()
