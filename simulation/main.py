from simulation.experiments.executable_experiments.agv_random_task_scheduling import run
from simulation.experiments_utils.test_graphs_builders import *
from simulation.core.queue_optimizer import QueueOptimizer  # Import the GNN-based QueueOptimizer
from simulation.core.tasks_scheduler import TasksScheduler  # Import the modified TasksScheduler
from simulation.core.traffic_controller import TrafficController  # Import the modified TrafficController
from model.gnn_model import GNNModel  # Import the GNN model
from simulation.core.system import System

# Initialize the system
graph_data = DebugGraphBuilder.buildGraph()  # Example graph initialization; change as per your requirement
system = System(graph_data=graph_data)

# Initialize GNN Model
gnn_model = GNNModel(num_node_features=system.graph_data.num_features, num_classes=1)
gnn_model.eval()  # Set GNN model to evaluation mode

# Run Different Simulations with GNN Integration
queue_optimizer = QueueOptimizer(system=system, agentsFactory=None, simulation=None, traverserFactory=None, queue=None, executorsManager=None)
tasks_scheduler = TasksScheduler(executorsManager=None, queueOptimizer=queue_optimizer)
traffic_controller = TrafficController(system=system)

# Replace the old `run()` function calls with GNN-integrated versions:
run(tasksNumber=100, agvsNumber=25, stationsNumber=40, graphBuilderClass=DebugGraphBuilder, subdirectory='debug')
run(tasksNumber=100, agvsNumber=50, stationsNumber=30, graphBuilderClass=VeryLongServiceTimeFullGraphBuilder, subdirectory='long_service_many_agvs')
run(tasksNumber=100, agvsNumber=12, stationsNumber=20, graphBuilderClass=LongServiceTimeFullGraphBuilder, subdirectory='long_service_few_agvs')
run(tasksNumber=100, agvsNumber=50, stationsNumber=10, graphBuilderClass=ShortServiceTimeFullGraphBuilder, subdirectory='short_service_many_agvs')

ShortServiceTimeFullGraphWithBranchesBuilder.branches = 1
run(tasksNumber=100, agvsNumber=50, stationsNumber=10, graphBuilderClass=ShortServiceTimeFullGraphWithBranchesBuilder, subdirectory='short_service_many_agvs_with_branches_k1')
ShortServiceTimeFullGraphWithBranchesBuilder.branches = 3
run(tasksNumber=100, agvsNumber=50, stationsNumber=10, graphBuilderClass=ShortServiceTimeFullGraphWithBranchesBuilder, subdirectory='short_service_many_agvs_with_branches_k3')
e_many_agvs_with_branches_k3')