import threading
from model.gnn_model import GNNModel  # Import the GNN model
import torch

LOCK_RANGE = 5

class TrafficController:
    def __init__(self, system):
        self.__system = system
        self.__lock = threading.Lock()

        # Initialize the GNN model for path optimization
        self.__gnn_model = GNNModel(num_node_features=system.graph_data.num_features, num_classes=1)
        self.__gnn_model.eval()  # Set to evaluation mode for inference

    def requestPath(self, source, destination, executor):
        k = 3
        sourceNode = self.__system.node(source)
        destinationNode = self.__system.node(destination)
        with self.__lock:
            # Get k shortest paths
            paths = self.__system.graph.get_k_shortest_paths(sourceNode.index, destinationNode.index, k=k)

            # GNN-based Path Selection
            optimized_path = self.__getOptimizedPath(paths)

            # Fall back to original path selection if GNN fails to provide a valid path
            if optimized_path is None:
                path = self.__pickFreePath(paths)
                if path is None:
                    path = self.__pickPartiallyFreePath(paths)
            else:
                path = optimized_path

            if path is not None:
                self.__assignSegment(path, executor, 0, LOCK_RANGE)
                return path
        return None

    def __getOptimizedPath(self, paths):
        # GNN inference to select the best path from the given paths
        with torch.no_grad():
            best_path = None
            best_score = float('-inf')
            for path in paths:
                # Prepare tensor representation of the path
                path_tensor = self._prepare_path_tensor(path)
                
                # Use GNN model to predict path score
                path_score = self.__gnn_model(path_tensor)
                
                if path_score.item() > best_score:
                    best_path = path
                    best_score = path_score.item()

            return best_path

    def _prepare_path_tensor(self, path):
        """
        Converts a path into a tensor representation suitable for the GNN model.
        Assumes that `path` is a list of nodes.
        """
        node_features = []
        for node in path:
            # Assuming each node has a feature vector stored in the graph
            node_feature = self.__system.graph.vs[node].get('features', [1.0])
            node_features.append(node_feature)
        
        path_tensor = torch.tensor(node_features, dtype=torch.float)
        return path_tensor

    def requestNextSegment(self, path, executor, startingPoint):
        with self.__lock:
            self.__unassignSegment(path, executor, startingPoint - 1, LOCK_RANGE)
            if self.__segmentFree(path, startingPoint, LOCK_RANGE):
                self.__assignSegment(path, executor, startingPoint, LOCK_RANGE)
                return True
        return False

    def segmentNodes(self, path, startingPoint):
        return self.__segmentNodes(path, startingPoint, LOCK_RANGE)

    def revokePath(self, path, executor):
        with self.__lock:
            self.__unassignSegment(path, executor, 0, len(path))

    def lowestCost(self, source, destination):
        path = self.__system.graph.get_k_shortest_paths(source, destination, k=1)[0]
        cost = 0
        i = 1
        while i < len(path):
            cost += self.__system.graph[path[i - 1], path[i]]
            i += 1
        return cost

    def isValidLocation(self, location):
        return location in self.__system.graph.vs.indices

    def __pickFreePath(self, paths):
        for path in paths:
            if self.__freePath(path):
                return path
        return None

    def __pickPartiallyFreePath(self, paths):
        for path in paths:
            if self.__partiallyFreePath(path):
                return path
        return None

    def __freePath(self, path):
        i = 1
        while i < len(path):
            executors = self.__system.edgeExecutors(path[i - 1], path[i])
            if len(executors) > 0:
                return False
            i += 1
        return True

    def __partiallyFreePath(self, path):
        return self.__segmentFree(path, 0, LOCK_RANGE)

    def __segmentFree(self, path, startingPoint, endingPoint):
        i = startingPoint + 1
        while i < len(path):
            executors = self.__system.edgeExecutors(path[i - 1], path[i])
            if len(executors) > 0:
                return False
            if (i - startingPoint - 1) == endingPoint:
                break
            i += 1
        return True

    def __segmentNodes(self, path, startingPoint, endingPoint):
        i = startingPoint
        segmentNodes = []
        while i < len(path):
            segmentNodes.append(path[i])
            i += 1
            if (i - startingPoint) == endingPoint:
                break
        return segmentNodes

    def __assignSegment(self, path, executor, startingPoint, endingPoint):
        i = startingPoint + 1
        while i < len(path):
            segmentExecutors = self.__system.edgeExecutors(path[i - 1], path[i])
            if len(segmentExecutors) > 1:
                raise Exception("Segment assigned to multiple agents!")
            segmentExecutors[id(executor)] = executor
            i += 1
            if (i - startingPoint - 1) == endingPoint:
                break

    def __unassignSegment(self, path, executor, startingPoint, endingPoint):
        i = startingPoint + 1
        while i < len(path):
            segmentExecutors = self.__system.edgeExecutors(path[i - 1], path[i])
            if id(executor) in segmentExecutors:
                del segmentExecutors[id(executor)]
            i += 1
            if (i - startingPoint - 1) == endingPoint:
                break
