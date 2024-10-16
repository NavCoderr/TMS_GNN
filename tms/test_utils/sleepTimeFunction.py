import random

avg = 1.1


def sleepFunction(mean):
    randomFactor = 0.5
    return (mean * (1-randomFactor)) + random.expovariate(1/(mean*randomFactor))


def check(samples):
    res = []

    for i in range(0, samples):
        res.append(sleepFunction(avg))

    print("Min: {}".format(min(res)))
    print("Avg: {}".format(sum(res)/len(res)))
    print("Max: {}".format(max(res)))

#check(100000000)