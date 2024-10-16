from simulation.experiments_utils.data_collectors.statistics_collector import StatisticsCollector


class RetriesCollector:
    def __init__(self, partialResultsObserver):
        self.__statisticsCollectors = list()
        self.__partialResultsObserver = partialResultsObserver
        self.__aggregateFunction = None

    def statisticsCollector(self):
        self.__statisticsCollectors.append(StatisticsCollector())
        return self.__statisticsCollectors[-1]

    def statistics(self, statistic):
        #future: handle aggregateFunction
        res = list()
        for statisticsCollector in self.__statisticsCollectors:
            res.append(statisticsCollector.avg(statistic))
        return res

    def onRetryFinished(self):
        self.__partialResultsObserver.onPartialResult(self.__statisticsCollectors[-1].avg('cost'))
