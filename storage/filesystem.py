class Filesystem:
    def __init__(self):
        self.__spooledFiles = {}

    def addFile(self, path, file):
        self.__spooledFiles[path] = file

    def readFile(self, path):
        if path in self.__spooledFiles:
            return self.__spooledFiles[path].read().decode('utf-8')
        with open(path, 'r') as f:
            return f.read()