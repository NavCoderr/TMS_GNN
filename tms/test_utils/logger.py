class Logger:
    def __init__(self, filename):
        self.__filename = filename

    def logLine(self, line):
        with open(self.__filename, 'a') as f:
            f.writelines("{}\n".format(line))
