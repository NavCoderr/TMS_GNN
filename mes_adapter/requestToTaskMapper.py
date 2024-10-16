import copy
from simulation.core.task import Task


testRequest2TaskMapping = {
    23: Task(23, 0, 1)
}


class RequestToTaskMapper:
    def __init__(self, configuration):
        self.__mapping = dict()
        self.__createMapping(configuration)

    def getTaskFromId(self, id, taskId):
        try:
            task = copy.deepcopy(self.__mapping[id])
            task.setTaskId(taskId)
            return task
        except KeyError:
            return None

    def __createMapping(self, configuration):
        for orderDefintion in configuration.ordersDefintions():
            self.__mapping[orderDefintion.orderId] = Task(orderDefintion.orderId, orderDefintion.source, orderDefintion.destination)