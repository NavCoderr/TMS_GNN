class Path:
    def __init__(self, path, cost, collisions, timeInQueue, timeInPenalty, timeInTransition):
        self.path = path
        self.cost = cost
        self.collisions = collisions
        self.timeInQueue = timeInQueue
        self.timeInPenalty = timeInPenalty
        self.timeInTransition = timeInTransition

    def __gt__(self, other):
        return self.cost > other.cost

    def __lt__(self, other):
        return other > self

    def __eq__(self, other):
        return not self > other and not other > self
