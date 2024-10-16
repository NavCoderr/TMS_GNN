import time
from dataclasses import dataclass

# number of seconds for which we consider the current state as valid
CACHE_INVALIDATION_PERIOD = 2


@dataclass
class CachedAgvState:
    status: object
    lastUpdateTime: object


class AgvStateCache:
    def __init__(self):
        self.__agvStateById = dict()
        self.__agvControllerClient = None

    def getAgvStatus(self, agvId):
        if agvId in self.__agvStateById:
            return self.__agvStateById[agvId].status
        else:
            return None

    def setClient(self, agvControllerClient):
        self.__agvControllerClient = agvControllerClient

    def cleanupAgvState(self, agvId):
        del self.__agvStateById[agvId]

    def updateAgvState(self, agvId):
        if agvId not in self.__agvStateById or self.__refreshRequired(agvId):
            self.__updateAgvState(agvId)
        return self.getAgvStatus(agvId)

    def __updateAgvState(self, agvId):
        if self.__agvControllerClient is not None:
            status = self.__agvControllerClient.requestAgvStatus(agvId)
            if status is not None:
                self.__agvStateById[agvId] = CachedAgvState(status, time.time())

    def __refreshRequired(self, agvId):
        return (time.time() - self.__agvStateById[agvId].lastUpdateTime) >= CACHE_INVALIDATION_PERIOD