import copy, threading


class TasksQueueView:
    def __init__(self, queue):
        self.queue = queue

    def cost(self):
        return self.queue.cost()

    def tasksList(self):
        return self.queue.tasksList()

    def pendingTasksList(self):
        return self.queue.pendingTasksList()


class TasksQueue:
    def __init__(self):
        self.__queue = list()
        self.__pendingTasks = list()
        self.__cost = -1
        self.__lock = threading.Lock()

    def enqueue(self, task):
        with self.__lock:
            self.__pendingTasks.append(task)

    def batchEnqueue(self, tasks):
        with self.__lock:
            self.__pendingTasks.extend(tasks)

    def onOptimizationStart(self):
        with self.__lock:
            self.__queue.extend(self.__pendingTasks)
            self.__pendingTasks = list()

    def onOptimizationFeedback(self, newSequence, cost):
        self.__validateNewSequence(newSequence)
        self.__queue = newSequence
        self.__cost = cost

    def onOptimizationFinished(self):
        pass

    def __validateNewSequence(self, newSequence):
        if len(newSequence) != len(self.__queue):
            raise Exception("Queue corruption during optimization, old: {}, new: {}".format(len(self.__queue), len(newSequence)))
        tmp = copy.deepcopy(newSequence)
        for task in self.__queue:
            index = -1
            for i in range(0, len(tmp)):
                if task.taskNumber() == tmp[i].taskNumber():
                    index = i
                    break
            if index == -1:
                raise Exception("Queue corruption during optimization, missing item: {}".format(task.taskNumber()))
            else:
                tmp.pop(index)

    def tasksList(self):
        return copy.deepcopy(self.__queue)

    def pendingTasksList(self):
        return copy.deepcopy(self.__pendingTasks)

    def size(self):
        return len(self.__queue)

    def queueView(self):
        return TasksQueueView(self)

    def popTask(self):
        return self.__queue.pop(0)

    def nextTask(self):
        return self.__queue[0]

    def empty(self):
        return len(self.__queue) == 0

    def cost(self):
        return self.__cost