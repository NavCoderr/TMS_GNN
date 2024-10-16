from dataclasses import dataclass
from simulation.core.agents_factory import AgentsFactory
from simulation.core.tasks_queue import TasksQueue, TasksQueueView
from simulation.core.traverser_base import TraverserStatistics
from model.gnn_model import GNNModel  # Import the GNN model
import torch

@dataclass
class OptimizationResult:
    queueView: TasksQueueView
    statistics: TraverserStatistics

class QueueOptimizer:
    def __init__(self, system, agentsFactory: AgentsFactory, simulation, traverserFactory, queue: TasksQueue, executorsManager):
        self.__system = system
        self.__agentsFactory = agentsFactory
        self.__simulation = simulation
        self.__traverserFactory = traverserFactory
        self.__traverser = None
        self.__queue = queue
        self.__executorsManager = executorsManager

        # Initialize the GNN model
        self.__gnn_model = GNNModel(num_node_features=system.graph_data.num_features, num_classes=len(queue.tasksList()))
        self.__gnn_model.eval()  # Set to evaluation mode for inference

    def optimizeQueue(self, iterations) -> OptimizationResult:
        executorsNumber = self.__executorsManager.onlineExecutorsNumber()
        self.__traverser = self.__traverserFactory(self.__system)
        
        if executorsNumber > 0:
            self.__queue.onOptimizationStart()
            tasksToOptimize = self.__queue.tasksList()
            oldSize = len(tasksToOptimize)

            if len(tasksToOptimize) > 1:
                # GNN-based Optimization Logic
                with torch.no_grad():
                    # Assuming that graph_data is a torch_geometric Data object containing the graph structure
                    graph_data = self.__system.graph_data
                    optimized_tasks_tensor = self.__gnn_model(graph_data)  # Get the optimized output from GNN
                    
                    # Process GNN output to get optimized sequence
                    optimized_task_indices = torch.argsort(optimized_tasks_tensor, descending=True)
                    optimizedSequence = [tasksToOptimize[idx] for idx in optimized_task_indices]

                # Integrate Multi-Path Search using `get_k_shortest_paths()`
                for task in optimizedSequence:
                    source, destination = task.getSource(), task.getDestination()
                    # Get multiple paths between source and destination
                    k_shortest_paths = self.__system.graph.get_k_shortest_paths(source, destination, k=3)

                    # Use GNN to evaluate and pick the best path
                    best_path = None
                    min_cost = float('inf')
                    for path in k_shortest_paths:
                        path_tensor = self._prepare_path_tensor(path)  # Assuming method to convert path to tensor
                        predicted_cost = self.__gnn_model(path_tensor)  # Predict cost with GNN
                        
                        if predicted_cost < min_cost:
                            min_cost = predicted_cost
                            best_path = path
                    
                    # Assign the best path to the task
                    task.setOptimizedPath(best_path)

                newSize = len(optimizedSequence)
                if oldSize != newSize:
                    raise Exception("Queue corrupted by optimizer, old: {}, new: {}!".format(oldSize, newSize))

                # Provide feedback to the queue
                self.__queue.onOptimizationFeedback(optimizedSequence, self.__traverser.cost())

            self.__queue.onOptimizationFinished()
        
        return OptimizationResult(self.__queue.queueView(), self.__traverser.statistics())

    def queue(self):
        return self.__queue

    def __createAgent(self, traverser):
        return self.__agentsFactory.createAgent({'traverser': traverser})

    def _prepare_path_tensor(self, path):
        """
        Converts a path into a tensor representation suitable for the GNN model.
        Assumes that `path` is a list of nodes.
        """
        # Placeholder logic: convert path into tensor.
        # You might want to encode the sequence of nodes as input features.
        node_features = []
        for node in path:
            # Assuming each node has a feature vector, e.g., extracted from the graph
            node_feature = self.__system.graph.vs[node].get('features', [1.0])
            node_features.append(node_feature)
        
        path_tensor = torch.tensor(node_features, dtype=torch.float)
        return path_tensor
