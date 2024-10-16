from simulation.experiments_utils.data_collectors.experiment_collector import ExperimentCollector


def mean(list):
    return sum(list) / len(list)


class DataSeries:
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values


class ExperimentAnalyzer:
    def __init__(self, experimentCollector : ExperimentCollector):
        self.__dataSource = experimentCollector
        self.__seriesFunctors = {
            'min': min,
            'mean': mean,
            'max': max
        }

    def analyze(self, statistic, seriesNames=None):
        res = dict()
        retriesByParameterValue = self.__dataSource.getStatisticsPerParameter(statistic)
        if seriesNames is None:
            seriesNames = self.getSupportedMeasures()
        for seriesName in seriesNames:
            res[seriesName] = self.__generateDataSeries(retriesByParameterValue, seriesName)
        return res

    def getSupportedMeasures(self):
        return self.__seriesFunctors.keys()

    def __generateDataSeries(self, retriesByParameterValue, seriesName):
        x_values = list()
        y_values = list()
        for parameterValue in retriesByParameterValue:
            x_values.append(parameterValue)
            functor = self.__seriesFunctors[seriesName]
            y_values.append(functor(retriesByParameterValue[parameterValue]))
        return DataSeries(x_values, y_values)