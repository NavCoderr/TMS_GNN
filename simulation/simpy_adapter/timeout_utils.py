import random


def timeoutFor(val):
#    return val
    return max(random.gauss(val, 0.5), 0.001)


def transitionTimeout(val):
#    return val
    return max(random.gauss(val, 0.5), 0.001)