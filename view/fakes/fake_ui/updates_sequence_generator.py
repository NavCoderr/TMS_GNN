from model.entities.visobject import VisObject, VisObjectData
from view.fakes.fake_events_hub import FakeObject


class UpdatesSequenceGenerator:
    def __init__(self):
        self.__abstractView = None
        self.__objects = dict()
        self.__modelMap = None
        self.__errorsListeners = list()

    def initialize(self, abstractView, objectsList, modelMap):
        self.__abstractView = abstractView
        self.__modelMap = modelMap
        for objectId in objectsList:
            self.__objects[objectId] = FakeObject(objectId, self.__modelMap.x(), self.__modelMap.y())

    def start(self):
        pass

    def stop(self):
        pass

    def addErrorListener(self, listener):
        self.__errorsListeners.append(listener)

    def updateObjectPosition(self, objectId, newX, newY):
        __object = self.__objects[objectId]
        __object.x = newX
        __object.y = newY
        objectToUpdate = VisObject(VisObjectData('dummy',
                                                 __object.objectId,
                                                 __object.x,
                                                 __object.y,
                                                 __object.rotation,
                                                 __object.width,
                                                 __object.height,
                                                 __object.properties()))
        self.__abstractView.renderObject(objectToUpdate)