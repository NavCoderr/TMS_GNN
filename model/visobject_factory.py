from model.entities.agv import *


class VisObjectFactory:
    def __init__(self):
        pass

    def createAgvObject(self, agvObjectData):
        return AgvObject(agvObjectData)

