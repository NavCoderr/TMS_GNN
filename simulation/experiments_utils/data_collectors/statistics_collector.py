class StatisticsCollector:
    def __init__(self):
        self.__statistics = dict()

    def collect(self, statistic, value):
        if statistic not in self.__statistics:
            self.__statistics[statistic] = list()
        self.__statistics[statistic].append(value)

    def avg(self, statistic):
        return sum(self.__statistics[statistic]) / len(self.__statistics[statistic])

    def statistic(self, statistic):
        return self.__statistics[statistic][-1]


