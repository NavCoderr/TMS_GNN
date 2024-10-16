class Edge:
    def __init__(self, source, destination, weight):
        self.source = source
        self.destination = destination
        self.weight = weight


class Graph:
    def __init__(self):
        self.vs = dict()  # Stores vertices and associated data
        self.es = []      # Stores edges

    def add_edges(self, edges):
        """
        Adds edges to the graph.
        :param edges: List of Edge objects to add
        """
        for edge in edges:
            self.es.append(edge)
            # Add edge to adjacency list representation
            if edge.source not in self.vs:
                self.vs[edge.source] = {'edges': []}
            if edge.destination not in self.vs:
                self.vs[edge.destination] = {'edges': []}
            self.vs[edge.source]['edges'].append(edge)

    def add_vertex(self, node):
        """
        Adds a vertex to the graph.
        :param node: Node data to add
        """
        i = len(self.vs)
        self.vs[i] = dict()
        self.vs[i]['node'] = node
        self.vs[i]['edges'] = []  # Initialize empty list of edges for the new node

    def __getitem__(self, item):
        """
        Retrieves weight between two nodes if edge exists.
        :param item: Tuple of (source, destination)
        :return: Weight of the edge or None if edge doesn't exist
        """
        source, destination = item
        for edge in self.es:
            if edge.source == source and edge.destination == destination:
                return edge.weight
        return None

    def __setitem__(self, key, value):
        """
        Adds or updates weight between two nodes.
        :param key: Tuple of (source, destination)
        :param value: Weight of the edge
        """
        source, destination = key
        for edge in self.es:
            if edge.source == source and edge.destination == destination:
                edge.weight = value
                return
        # If edge doesn't exist, create a new one
        new_edge = Edge(source, destination, value)
        self.add_edges([new_edge])

    def get_k_shortest_paths(self, s, d, k):
        """
        A stub for getting k shortest paths. Replace with actual pathfinding logic.
        :param s: Source node
        :param d: Destination node
        :param k: Number of shortest paths to find
        :return: List of paths
        """
        return [[s, d]]  # Placeholder

    def get_adjacency_matrix(self):
        """
        Generates the adjacency matrix of the graph.
        :return: 2D list representing the adjacency matrix
        """
        num_nodes = len(self.vs)
        adjacency_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for edge in self.es:
            adjacency_matrix[edge.source][edge.destination] = edge.weight
            adjacency_matrix[edge.destination][edge.source] = edge.weight  # Assuming undirected graph
        return adjacency_matrix

    def get_node_features(self):
        """
        Extracts features of nodes for GNN.
        :return: Dictionary of node features keyed by node index.
        """
        node_features = {}
        for node_index, data in self.vs.items():
            # Placeholder: Extract relevant features. This might be more complex in a real scenario.
            node_features[node_index] = data.get('features', [1.0])  # Default to 1.0 if no features available
        return node_features

    def get_edge_list(self):
        """
        Generates edge list for GNN usage.
        :return: List of tuples representing edges (source, destination, weight).
        """
        return [(edge.source, edge.destination, edge.weight) for edge in self.es]
