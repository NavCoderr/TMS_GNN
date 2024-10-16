from model.entities.visobject import *


class AgvObjectData:
    def __init__(self, visObjectData):
        self.visObjectData = visObjectData


class AgvObject(VisObject):
    def __init__(self, agvObjectData):
        super().__init__(agvObjectData.visObjectData)
