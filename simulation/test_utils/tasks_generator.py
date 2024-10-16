import random
from simulation.core.task import Task


def generateRandomTask(nodesNumber):
    n1 = random.randint(0, nodesNumber - 1)
    n2 = random.randint(0, nodesNumber - 1)
    while n1 == n2:
        n2 = random.randint(0, nodesNumber - 1)
    return Task(source=n1, destination=n2, taskNumber=1)


def generateTasksQueue(len, nodesNumber):
    tasks = list()
    for i in range(0, len):
        tasks.append(generateRandomTask(nodesNumber))
    return tasks

