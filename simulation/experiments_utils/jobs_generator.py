import copy
import random


def generateRandomJobs(jobsNumber, nodesNumber):
    testJobs = dict()
    permutations = list()
    for i in range(0, nodesNumber):
        permutations.append(i)

    for i in range(0, jobsNumber):
        random.shuffle(permutations)
        testJobs[str(i)] = copy.deepcopy(permutations)

    return testJobs

