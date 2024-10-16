import matplotlib.pyplot as plt


dark_gray = '#444444'
dark_blue = '#6060C0'
dark_green = '#60C060'
dark_red = '#F06060'
dark_violet = '#A040A0'

pretty_colors = [dark_blue, dark_red, dark_green, dark_violet, dark_gray]


def plotSeries(seriesDict, title, ylabel, xlabel, filename, labelsBySeriesName):
    fig, ax = plt.subplots()

    for seriesName in seriesDict:
        ax.plot(seriesDict[seriesName].x_values, seriesDict[seriesName].y_values, linewidth=1, markersize=2, marker='o', label=labelsBySeriesName[seriesName])

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if filename != "":
        plt.savefig(filename)
    else:
        plt.show()


def plotStackedSeries(seriesDict, title, filename):
    fig, ax = plt.subplots()
    x = list(seriesDict.values())[0].x_values
    y = []
    labels = []
    for seriesName in seriesDict:
        y.append(seriesDict[seriesName].y_values)
        labels.append(seriesName)

    ax.stackplot(x, *y, labels=labels, colors=pretty_colors)
    ax.legend(loc='lower right')
    ax.set_title(title)
    if filename != "":
        plt.savefig(filename)
    else:
        plt.show()
