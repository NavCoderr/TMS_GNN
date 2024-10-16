import csv, os


class CsvWriter:
    def __init__(self, path, experimentAnalyzer):
        self.__path = path
        self.__analyzer = experimentAnalyzer

    def write(self, statistic, parameter, legend):
        path = self.__path
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        filepath = os.path.join(path, "{}.csv".format(statistic))
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            self.__writeData(csvwriter, statistic, parameter)
        legendFilepath = os.path.join(path, "legend.txt")
        with open(legendFilepath, 'w', newline='') as legendFile:
            self.__writeLegend(legend, legendFile)


    def __writeHeader(self, csvwriter, dataGrid):
        row = []
        for column in dataGrid:
            row.append(column)
        csvwriter.writerow(row)

    def __writeData(self, csvwriter, statistic, parameter):
        seriesDict = self.__analyzer.analyze(statistic)
        dataGrid = dict()
        for seriesName in seriesDict:
            dataGrid[parameter] = seriesDict[seriesName].x_values
            dataGrid[seriesName] = seriesDict[seriesName].y_values

        self.__writeHeader(csvwriter, dataGrid)
        rowsCount = len(dataGrid[parameter])
        for i in range(0, rowsCount):
            row = []
            for column in dataGrid:
                row.append(dataGrid[column][i])
            csvwriter.writerow(row)

    def __writeLegend(self, legend, legendFile):
        for name in legend:
            line = "{}: {}\n".format(name, legend[name])
            legendFile.write(line)

