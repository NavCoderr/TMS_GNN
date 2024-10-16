from simulation.simpy_adapter.timeout_utils import *


class Agent:

    def __init__(self, env, number, traverser, startingNode = None):
        self.env = env
        self.number = number
        self.traverser = traverser
        self.currentNode = startingNode
        self.nextNode = None

    def start(self):
        self.env.process(self.__run())


    def __run(self):
        startTime = self.env.now

        tasks = self.traverser.tasks()
        i = 1
        collisions = 0
        timeInQueue = 0
        timeInPenalty = 0
        timeInTransition = 0
        penaltyTime = 200
        while i <= len(tasks):
            task = tasks[i-1]

            if self.currentNode is not None and self.currentNode.index != task.source():
                sourceNode = self.traverser.node(task.source())

                beforeTransit = self.env.now
                path = self.traverser.pathBetweenNodes(self.currentNode, sourceNode)
                i = 1
                transitCollisionTime = 0
                while i < len(path):
                    transitionTime = self.traverser.transitionTime(path[i - 1], path[i])
                    agents = self.traverser.edgeAgents(path[i - 1], path[i])
                    agents[id(self)] = self
                    for agentId in agents:
                        if agentId != id(self):
                            beforePenalty = self.env.now
                            yield self.env.timeout(transitionTimeout(penaltyTime))
                            collisions += 1
                            timeInPenalty += (self.env.now - beforePenalty)
                            transitCollisionTime += (self.env.now - beforePenalty)
                            break

                    yield self.env.timeout(transitionTimeout(transitionTime))
                    del agents[id(self)]
                    i += 1

                afterTransit = self.env.now
                timeInTransition += afterTransit - beforeTransit - transitCollisionTime

            self.currentNode = self.traverser.node(task.source())
            self.nextNode = self.traverser.node(task.destination())

            currentNodeEnter = self.env.now
            self.currentNode.onEnqueue()
            with self.currentNode.executor.request() as request:
                yield request
                yield self.env.process(self.currentNode.startTask(task.taskNumber()))
            self.currentNode.onDeque()
            currentNodeLeaving = self.env.now
            timeInQueue += currentNodeLeaving - currentNodeEnter

            beforeTransit = self.env.now
            path = self.traverser.pathBetweenNodes(self.currentNode, self.nextNode)
            i = 1
            transitCollisionTime = 0
            while i < len(path):
                transitionTime = self.traverser.transitionTime(path[i - 1], path[i])
                agents = self.traverser.edgeAgents(path[i - 1], path[i])
                agents[id(self)] = self
                for agentId in agents:
                    if agentId != id(self):
                        beforePenalty = self.env.now
                        yield self.env.timeout(transitionTimeout(penaltyTime))
                        collisions += 1
                        timeInPenalty += (self.env.now - beforePenalty)
                        transitCollisionTime += (self.env.now - beforePenalty)
                        break

                yield self.env.timeout(transitionTimeout(transitionTime))
                del agents[id(self)]
                i += 1

            afterTransit = self.env.now
            timeInTransition += afterTransit - beforeTransit - transitCollisionTime

            nextNodeEnterTime = self.env.now
            self.nextNode.onEnqueue()
            with self.nextNode.executor.request() as request:
                yield request
                yield self.env.process(self.nextNode.startTask(task.taskNumber()))
            self.nextNode.onDeque()
            nextNodeLeaveTime = self.env.now
            timeInQueue += (nextNodeLeaveTime - nextNodeEnterTime)

            self.currentNode = self.nextNode

            i += 1

        endTime = self.env.now
        tasksCost = endTime - startTime
        self.traverser.feedback(tasksCost, collisions, timeInQueue, timeInPenalty, timeInTransition)
