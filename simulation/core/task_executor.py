class TaskExecutor:

    def execute(self, task, taskId):
        raise NotImplementedError()

    def getId(self):
        raise NotImplementedError()

    def getLocation(self):
        raise NotImplementedError()

    def isOnline(self):
        raise NotImplementedError()
