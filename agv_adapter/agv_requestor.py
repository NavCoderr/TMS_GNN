import threading
from dataclasses import dataclass


@dataclass
class Request:
    agvId: str
    points: list
    taskId: object


class AgvRequestor:
    def __init__(self):
        self.__lock = threading.Lock()
        self.__requestsQueue = list()
        self.__agvControllerClient = None

    def requestGoToPoints(self, agvId, points, taskId):
        with self.__lock:
            self.__requestsQueue.append(Request(agvId, points, taskId))

    def processRequests(self):
        if self.__agvControllerClient is None or len(self.__requestsQueue) == 0:
            return
        with self.__lock:
            for request in self.__requestsQueue:
                self.__agvControllerClient.requestGoToPoints(request.agvId, request.points, request.taskId)
            self.__requestsQueue.clear()

    def setClient(self, agvControllerClient):
        self.__agvControllerClient = agvControllerClient
