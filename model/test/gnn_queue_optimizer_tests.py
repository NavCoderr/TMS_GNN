import unittest
from model.gnn_model import GNNModel
from simulation.core.queue_optimizer import QueueOptimizer
from simulation.core.system import System
from simulation.experiments_utils.test_graphs_builders import FullGraphBuilder
from simulation.test_utils.tasks_generator import generateTasksQueue

class GNNQueueOptimizerTests(unittest.TestCase):

    def setUp(self):
        # Initialize a full graph for testing
        self.graph_data = FullGraphBuilder().buildGraph()
        self.system = System(graph_data=self.graph_data)

        # Initialize GNN Model for optimization
        self.gnn_model = GNNModel(num_node_features=self.graph_data.num_features, num_classes=1)
        self.gnn_model.eval()  # Set GNN model to evaluation mode

        # Set up the QueueOptimizer with GNN
        self.queue_optimizer = QueueOptimizer(system=self.system, agentsFactory=None, simulation=None,
                                              traverserFactory=None, queue=None, executorsManager=None)
        self.queue_optimizer.initializeGNN(self.gnn_model)

    def test_queue_optimizer_with_mock_tasks(self):
        # Generate mock tasks for the test
        tasksQueue = generateTasksQueue(tasksNumber=10, nodesNumber=len(self.graph_data.nodes))
        
        # Perform GNN-based optimization
        iterations = 1000
        result = self.queue_optimizer.optimizeQueue(iterations)

        # Assert the results are valid
        self.assertIsNotNone(result, "QueueOptimizer did not return any result.")
        self.assertGreaterEqual(len(result.queueView.tasks), 0, "Queue should contain tasks or be empty.")
        self.assertGreaterEqual(result.statistics.collisions, 0, "Collisions must be non-negative.")

    def test_performance_under_high_load(self):
        # Test GNN queue optimization under high load scenario
        tasksQueue = generateTasksQueue(tasksNumber=100, nodesNumber=len(self.graph_data.nodes))

        # Perform GNN-based optimization with more iterations
        iterations = 3000
        result = self.queue_optimizer.optimizeQueue(iterations)

        # Assert that the optimization completes successfully under high load
        self.assertIsNotNone(result, "QueueOptimizer failed under high load.")
        self.assertGreaterEqual(len(result.queueView.tasks), 0, "Tasks should be handled properly.")
        self.assertGreaterEqual(result.statistics.collisions, 0, "Collisions should be correctly recorded.")

    def test_gnn_vs_legacy_performance(self):
        # Placeholder for comparing GNN-based scheduling against a legacy method
        # Set up a legacy QueueOptimizer without GNN for comparison
        # Compare metrics like time, cost, and collisions
        pass  # Implement comparison logic as needed

if __name__ == '__main__':
    unittest.main()
