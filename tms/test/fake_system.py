class FakeSystem:
    def __init__(self):
        self.vertices=[]
        self.edges=[]

    def addVertex(self, vertex):
        self.vertices.append(vertex)

    def addEdge(self, edge):
        self.edges.append(edge)
