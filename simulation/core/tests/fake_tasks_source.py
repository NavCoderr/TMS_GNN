from simulation.core.tasks_source import TasksSource
from simulation.core.task import Task
from simulation.core.tasks_queue import TasksQueue


class FakeTasksSource(TasksSource):
    def __init__(self):
        self.__tasksQueue = None
        self.__tasks = list()

    def setTasksQueue(self, queue: TasksQueue):
        self.__tasksQueue = queue

    def addTask(self, task: Task):
        self.__tasks.append(task)

    def startProcessing(self, interval):
        for task in self.__tasks:
            self.__tasksQueue.enqueue(task)
