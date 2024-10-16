import random, time
from simulation.core.task_executor import TaskExecutor


fakeExecutors = dict()


class FakeTasksExecutor(TaskExecutor):
    def __init__(self):
        self.__task = None
        self.__observers = dict()
        fakeExecutors[id(self)] = self
        self.__executedTasks = 0

    def execute(self, task):
        self.__task = task
        self.__executedTasks += 1
        time.sleep(random.random()*0.2)

    def addObserver(self, executorObserver):
        self.__observers[id(executorObserver)] = executorObserver

    def getExecutedTasksNumber(self):
        return self.__executedTasks
