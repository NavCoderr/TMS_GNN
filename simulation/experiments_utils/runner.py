class Runner:
    def __init__(self, experiment, retriesCollector):
        self.__experiment = experiment
        self.__retriesCollector = retriesCollector

    def run(self, times):
        for i in range(0, times):
            self.__experiment.run(self.__retriesCollector.statisticsCollector())
            self.__retriesCollector.onRetryFinished()

