import csv, os
from simulation.experiments_utils.analytics.experiment_analyzer import DataSeries


class CsvReader:

    def __init__(self, path):
        self.__path = path

    def read(self, filename, series):
        rawData = dict()
        labels = list()
        filepath = os.path.join(self.__path, filename)
        x_label = ""
        with open(filepath, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';')
            firstRow = True
            for row in csvreader:
                if firstRow:
                    x_label = row[0]
                    for label in row:
                        rawData[label] = list()
                        labels.append(label)
                    firstRow = False
                else:
                    index = 0
                    for value in row:
                        rawData[labels[index]].append(float(value))
                        index += 1
        return self.__rawData2DataSeries(rawData, x_label, series)

    def __rawData2DataSeries(self, rawData, x_label, series):
        x_values = rawData[x_label]
        res = dict()
        for label in rawData:
            if label != x_label and label in series:
                res[label] = DataSeries(x_values=x_values, y_values=rawData[label])
        return res
