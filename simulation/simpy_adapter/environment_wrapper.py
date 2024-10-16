import simpy


class EnvironmentWrapper:
    def __init__(self, timeout):
        self.env = simpy.Environment()
        self.__timeout = timeout
        self.__curTime = 0

    def run(self):
        self.__curTime += self.__timeout
        self.env.run(until=self.__curTime)
