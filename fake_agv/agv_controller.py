class AgvController:
    def __init__(self):
        self.__agvs = dict()

    def addAgv(self, agvName, agv):
        self.__agvs[agvName] = agv