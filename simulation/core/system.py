try:
    from igraph import Graph
except ModuleNotFoundError:
    print("Please install igraph module: python -m pip install igraph")
    from graph.graph import Graph


class System:
    def __init__(self):
        self.graph = Graph()

    def node(self, index):
        return self.graph.vs[index]['node']

    def edgeWeight(self, source, destination):
        return self.graph[source, destination]

    def edgeAgents(self, source, destination):
        edge = self.graph.es[self.graph.get_eid(source, destination)]
        return edge['agents']

    def edgeExecutors(self, source, destination):
        edge = self.graph.es[self.graph.get_eid(source, destination)]
        return edge['executors']

    def nodesCount(self):
        return len(self.graph.vs)