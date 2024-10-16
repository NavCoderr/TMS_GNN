import time, threading, inspect
from simulation.core.tasks_executor_manager import TasksExecutorManager
from agv_adapter.agv_task_executor import AgvTaskExecutor
from agv_adapter.agv_controller_client import AgvControllerClient
from agv_adapter.agv_state_cache import AgvStateCache
from agv_adapter.agv_requestor import AgvRequestor


class AgvTaskExecutorManager(TasksExecutorManager):
    def __init__(self, agvControllerIp, agvControllerPort):
        self.__agvTaskExecutors = dict()
        self.__ip = agvControllerIp
        self.__port = agvControllerPort
        self.__agvControllerClient = None
        self.__observers = dict()
        self.__killed = False
        self.__pollingThread = None
        self.__agvCache = AgvStateCache()
        self.__agvRequestor = AgvRequestor()

        self.__createRefreshThread()

    def tasksExecutors(self):
        return list(self.__agvTaskExecutors.values())

    def addTasksExecutorObserver(self, observer):
        self.__observers[id(observer)] = observer

    def removeTasksExecutorObserver(self, observer):
        del self.__observers[id(observer)]

    def kill(self):
        self.__killed = True

    def isClientRunning(self):
        return self.__agvControllerClient is not None and self.__agvControllerClient.connected()

    def onExecutorChanged(self):
        self.__broadcastExecutorsChanged()

    def __broadcastExecutorsChanged(self):
        for observerId in self.__observers:
            self.__observers[observerId].onTasksExecutorsChanged()

    def performRequests(self):
        self.__agvRequestor.processRequests()

    def refreshTasksExecutors(self):
        if not self.isClientRunning() or self.__agvControllerClient.busy():
            return

        self.__updateAvailableAgvs()
        self.__refreshExecutorsStatus()

    def __updateAvailableAgvs(self):
        newAvailableAgvIds = self.__agvControllerClient.requestAgvsIds()
        if newAvailableAgvIds is not None and newAvailableAgvIds != self.__availableAgvs():
            self.__unregisterUnavailableExecutors(newAvailableAgvIds)
            self.__registerNewAvailableExecutors(newAvailableAgvIds)
            self.__broadcastExecutorsChanged()

    def __unregisterUnavailableExecutors(self, availableAgvIds):
        executorsToCleanup = []
        for agvId in self.__agvTaskExecutors:
            if agvId not in availableAgvIds:
                executorsToCleanup.append(agvId)

        for agvId in executorsToCleanup:
            del self.__agvTaskExecutors[agvId]
            self.__agvCache.cleanupAgvState(agvId)

    def __registerNewAvailableExecutors(self, availableAgvIds):
        for agvId in availableAgvIds:
            agvStatus = self.__agvCache.updateAgvState(agvId)
            if agvStatus is not None:
                if agvId not in self.__agvTaskExecutors:
                    self.__agvTaskExecutors[agvId] = AgvTaskExecutor(agvId, self.__agvCache, self.__agvRequestor, self, agvStatus)

    def __refreshExecutorsStatus(self):
        for agvId in self.__agvTaskExecutors:
            try:
                self.__agvTaskExecutors[agvId].updateStatus(self.__agvCache.updateAgvState(agvId))
            except KeyError:
                pass

    def __refreshOfflineExecutors(self):
        for agvId in self.__agvTaskExecutors:
            try:
                if not self.__agvTaskExecutors[agvId].isOnline:
                    self.__agvTaskExecutors[agvId].updateStatus(self.__agvCache.updateAgvState(agvId))
            except KeyError:
                pass

    def __availableAgvs(self):
        return list(self.__agvTaskExecutors.keys())

    def __cleanupTasksExecutors(self):
        if len(self.__availableAgvs()) > 0:
            for agvId in self.__agvTaskExecutors:
                self.__agvTaskExecutors[agvId].kill()

            self.__agvTaskExecutors = dict()
            self.__broadcastExecutorsChanged()

    def __ensureClientRunning(self):
        if not self.isClientRunning():
            self.__setupAgvClient()
        if not self.isClientRunning():
            self.__cleanupTasksExecutors()

    def __setupAgvClient(self):
        # since connecting to AGV Controller may take a while, clean up "old" client which will soon become invalid
        # so the other threads do not user corrupted client
        self.__agvCache.setClient(None)
        self.__agvRequestor.setClient(None)
        # update the client
        self.__agvControllerClient = AgvControllerClient(self.__ip, self.__port)
        if self.isClientRunning():
            # if connection succeeded, inject the dependencies
            self.__agvCache.setClient(self.__agvControllerClient)
            self.__agvRequestor.setClient(self.__agvControllerClient)

    def __reconnect(self):
        while not self.isClientRunning():
            self.__ensureClientRunning()
            time.sleep(5)
            if self.__killed:
                break

    def __createRefreshThread(self):
        self.__pollingThread = threading.Thread(target=self.__refreshConnection)
        self.__pollingThread.daemon = True
        self.__pollingThread.start()

    def __refreshConnection(self):
        while not self.__killed:
            time.sleep(5)
            self.__reconnect()