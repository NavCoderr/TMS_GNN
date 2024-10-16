from simulation.experiments_utils.data_collectors.retries_collector import RetriesCollector


class ExperimentCollector:
    def __init__(self, partialResultsObserver):
        self.__retriesPerParameterValue = dict()
        self.__partialResultsObserver = partialResultsObserver

    def getRetriesCollector(self, parameterValue):
        collector = RetriesCollector(self.__partialResultsObserver)
        self.__retriesPerParameterValue[parameterValue] = collector
        return collector

    def getStatisticsPerParameter(self, statistic):
        res = dict()
        for parameterValue in self.__retriesPerParameterValue:
            res[parameterValue] = self.__retriesPerParameterValue[parameterValue].statistics(statistic)
        return res
