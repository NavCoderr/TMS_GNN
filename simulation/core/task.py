class Task:
    def __init__(self, taskNumber, source, destination, taskId = -1):
        self.__taskNumber = taskNumber
        self.__source = source
        self.__destination = destination
        self.__taskId = taskId

    def setTaskId(self, taskId):
        self.__taskId = taskId

    def source(self):
        return self.__source

    def destination(self):
        return self.__destination

    def taskNumber(self):
        return self.__taskNumber

    def __str__(self):
        return "{}: [{}, {}]".format(self.__taskNumber, self.__source, self.__destination)

    def pointsSequence(self):
        return [self.source(), self.destination()]

    def taskId(self):
        return self.__taskId