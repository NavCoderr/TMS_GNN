import simpy, copy
from simulation.simpy_adapter.timeout_utils import *


class Node:
    def __init__(self, env, serviceTime, index):
        self.env = env
        self.executor = simpy.Resource(env, 1)
        self.serviceTime = serviceTime
        self.index = index
        self.__agentsLeaving = dict()
        self.__queueLengths = list()
        self.__currentQueueLength = 0

    def startTask(self, taskNumber):
        yield self.env.timeout(timeoutFor(self.serviceTime))

    def endTask(self, taskNumber):
        yield self.env.timeout(timeoutFor(self.serviceTime))

    def addAgentLeavingNode(self, agent):
        self.__agentsLeaving[id(agent)] = agent

    def removeAgentLeavingNode(self, agent):
        del self.__agentsLeaving[id(agent)]

    def getAgentsLeavingNode(self):
        return self.__agentsLeaving.copy()

    def onEnqueue(self):
        self.__currentQueueLength += 1

    def onDeque(self):
        self.__currentQueueLength -= 1
        self.__queueLengths.append(self.__currentQueueLength)

    def queueLengthHistory(self):
        return self.__queueLengths