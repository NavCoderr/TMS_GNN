import copy, time
from simulation.core.composition_root import CompositionRoot as CoreRoot
from simulation.core.composition_root import SimulationInitInfo
from simulation.simpy_adapter.composition_root import CompositionRoot as SimpyRoot
from simulation.core.tasks_executor_manager import TasksExecutorManager
from simulation.core.task_executor import TaskExecutor
from simulation.core.queue_optimizer import QueueOptimizer  # Import GNN-enhanced QueueOptimizer
from model.gnn_model import GNNModel  # Import GNN Model

class FakeTaskExecutor(TaskExecutor):
    def __init__(self, executorId):
        self.__executorId = executorId

    def execute(self, jobId):
        pass

    def getId(self):
        return self.__executorId


class FakeTasksExecutorsManager(TasksExecutorManager):
    def __init__(self, executorsNumber):
        self.__executorsNumber = executorsNumber
        self.__executors = []
        for i in range(self.__executorsNumber):
            self.__executors.append(FakeTaskExecutor(i))

    def tasksExecutors(self):
        return self.__executors

    def addTasksExecutorObserver(self, observer):
        pass

    def removeTasksExecutorObserver(self, observer):
        pass


class RandomTasksScheduling:

    def __init__(self, tasksQueue, executorsNumber, iterations, builder, traverserName):
        self.__coreRoot = CoreRoot()
        self.__simpyRoot = SimpyRoot(10000000)
        self.__tasksQueue = tasksQueue
        self.__iterations = iterations
        self.__testGraphBuilder = builder.setEnvironment(self.__simpyRoot.simulation.env)
        self.__traverserName = traverserName
        self.__executorsManager = FakeTasksExecutorsManager(executorsNumber)
        
        # Initialize GNN Model
        graph_data = self.__testGraphBuilder.buildGraph()  # Build the graph data
        self.__gnn_model = GNNModel(num_node_features=graph_data.num_features, num_classes=1)
        self.__gnn_model.eval()  # Set GNN model to evaluation mode

        # Initialize GNN-based QueueOptimizer
        self.__queue_optimizer = QueueOptimizer(system=self.__coreRoot.system(), agentsFactory=None, 
                                                simulation=self.__simpyRoot.simulation, traverserFactory=None, 
                                                queue=self.__tasksQueue, executorsManager=self.__executorsManager)
        self.__queue_optimizer.initializeGNN(self.__gnn_model)

    def run(self, statisticsCollector):
        dependencies = {'agentsFactory': self.__simpyRoot.simpyAgentsFactory,
                        'simulation': self.__simpyRoot.simulation,
                        'taskExecutorsManager': self.__executorsManager}
        simulationInitInfo = SimulationInitInfo(traverserName=self.__traverserName)
        self.__coreRoot.initialize(dependencies, self.__testGraphBuilder, simulationInitInfo)
        self.__coreRoot.tasksQueue().batchEnqueue(copy.deepcopy(self.__tasksQueue))
        
        t1 = time.time()
        # Use GNN-enhanced QueueOptimizer for task scheduling
        res = self.__queue_optimizer.optimizeQueue(self.__iterations)
        t2 = time.time()
        
        elapsedTime = t2 - t1
        statisticsCollector.collect('time', elapsedTime)
        statisticsCollector.collect('cost', res.queueView.cost())
        statisticsCollector.collect('collisions', res.statistics.collisions)
        statisticsCollector.collect('timeInQueue', res.statistics.timeInQueue)
        statisticsCollector.collect('timeInPenalty', res.statistics.timeInPenalty)
        statisticsCollector.collect('timeInTransition', res.statistics.timeInTransition)

        for i in range(0, self.__coreRoot.system().nodesCount()):
            tmp = self.__coreRoot.system().node(i).queueLengthHistory()
            if len(tmp) > 0:
                statisticsCollector.collect('queueLength', sum(tmp) / len(tmp))
