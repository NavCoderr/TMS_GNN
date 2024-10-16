from simulation.experiments_utils.plotters.boxplot import plotSeries
from simulation.experiments_utils.csv_reader import CsvReader


def plot(path, file, columns, title, y_label, x_label, plotFilename):
    reader = CsvReader(path)

    dataSeries = reader.read(file, columns)
    plotSeries(dataSeries, title, y_label, x_label, plotFilename)

path = '/simulation/experiments/results_bkp/agv_random_task_scheduling/long_service_few_agvs'
file = 'cost.csv'

plot(path, file, ['mean'], 'Average path cost', 'Cost', 'Calculation time [iterations]', '')