import copy
from simulation.core.task import Task
from simulation.core.task_executor import TaskExecutor
import threading, time


class JobExecutor:
    def __init__(self, actualExecutor: TaskExecutor, owner):
        self.__job = None
        self.__taskExecutor = actualExecutor
        self.__currentTask = 0
        self.__pathPoint = 0
        self.__busy = False
        self.__owner = owner
        self.__thread = None
        self.__state = "idle"
        self.__path = None
        self.__killed = False
        self.__remainingJob = None

    def busy(self):
        return self.__busy

    def online(self):
        return self.__taskExecutor.isOnline()

    def availableForJobs(self):
        return self.online() and not self.busy()

    def executeJob(self, job):
        self.__remainingJob = None
        self.__killed = False
        self.__busy = True
        self.__state = "assigned"
        self.__currentTask = 0
        self.__job = job
        self.__thread = threading.Thread(target=self.__executeJob)
        self.__thread.daemon = True
        self.__thread.start()

    def kill(self):
        self.__killed = True
        if not self.__killed:
            self.__thread.join()
            self.__thread = None
            self.__unassignJob()

    def tasksCount(self):
        if self.__job is None:
            return 0
        return len(self.__job) - self.__currentTask

    def taskExecutorId(self):
        return self.__taskExecutor.getId()

    def job(self):
        return self.__job

    def currentTask(self):
        return self.__currentTask

    def assignedPath(self):
        return self.__path

    def state(self):
        if not self.online():
            return "offline"
        return self.__state

    def pathPoint(self):
        return self.__pathPoint

    def remainingJob(self):
        if self.__remainingJob is not None:
            return self.__remainingJob
        if self.__job is not None:
            return self.__job[self.currentTask():]
        return []

    def location(self):
        return int(self.__taskExecutor.getLocation())

    def __executeJob(self):
        try:
            if self.__goToSourceLocation():
                for i in range(0, len(self.__job)):
                    self.__currentTask = i
                    if not self.__executeTask(self.__job[self.__currentTask]):
                        self.__backupRemainingJob()
                        break
            self.__onJobFinished()
        except Exception as e:
            print("Unexpected exception!: {}".format(str(e)), flush=True)

    def __executeTask(self, task):
        if self.__killed:
            return False

        points = task.pointsSequence()
        self.__path = self.__waitForFreePath(points[0], points[1])
        self.__pathPoint = 0

        for _ in self.__path:
            self.__state = "running"
            succeeded = self.__taskExecutor.execute(self.__currentSegmentNodes(), task.taskId())
            if self.__killed or not succeeded:
                return False

            self.__pathPoint += 1
            self.__waitForFreeSegment(self.__path, self.__pathPoint)

        self.__owner.trafficController().revokePath(self.__path, self)
        return True

    def __goToSourceLocation(self):
        if self.location() != self.__job[0].source():
            dummyTaskId = -1
            return self.__executeTask(Task(dummyTaskId, self.location(), self.__job[0].source(), self.__job[0].taskId()))
        return True

    def __onJobFinished(self):
        self.__unassignJob()
        self.__owner.onExecutorFinished()

    def __unassignJob(self):
        if self.__path is not None:
            self.__owner.trafficController().revokePath(self.__path, self)
        self.__state = "idle"
        self.__job = None
        self.__busy = False
        self.__path = None
        self.__pathPoint = 0
        self.__currentTask = 0

    def __waitForFreePath(self, source, destination):
        self.__state = "waiting_for_path"
        path = self.__owner.trafficController().requestPath(source, destination, self)
        while path is None:
            path = self.__owner.trafficController().requestPath(source, destination, self)
            time.sleep(1)
        return path

    def __waitForFreeSegment(self, path, startingPoint):
        self.__state = "waiting_for_path"
        while not self.__owner.trafficController().requestNextSegment(path, self, startingPoint):
            time.sleep(1)

    def __currentSegmentNodes(self):
        return self.__owner.trafficController().segmentNodes(self.__path, self.__pathPoint)

    def __backupRemainingJob(self):
        self.__remainingJob = self.remainingJob()


class JobExecutorView:
    def __init__(self, jobExecutor: JobExecutor):
        self.__executor = jobExecutor

    def tasksCount(self):
        return self.__executor.tasksCount()

    def executorId(self):
        return self.__executor.taskExecutorId()

    def tasksSequence(self):
        return self.__executor.remainingJob()

    def assignedPath(self):
        path = self.__executor.assignedPath()
        if path is not None:
            return path
        return []

    def pathPoint(self):
        return self.__executor.pathPoint()

    def state(self):
        return self.__executor.state()
