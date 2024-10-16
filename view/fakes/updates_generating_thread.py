import threading
import time
from model.entities.visobject import VisObject, VisObjectData
from view.fakes.fake_events_hub import FakeObject


class UpdatesGeneratingThread:
    def __init__(self, generationInterval):
        self.__thread = None
        self.__abstractView = None
        self.__objectsList = list()
        self.__stopped = False
        self.__generationInterval = generationInterval
        self.__modelMap = None
        self.__errorsListeners = list()

    def initialize(self, abstractView, objectsList, modelMap):
        self.__abstractView = abstractView
        self.__modelMap = modelMap
        for objectId in objectsList:
            self.__objectsList.append(FakeObject(objectId, self.__modelMap.x(), self.__modelMap.y()))

    def start(self):
        self.__thread = threading.Thread(target=self.__generateUpdates)
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        self.__stopped = True
        self.__thread.join()
        self.__thread = None

    def addErrorListener(self, listener):
        self.__errorsListeners.append(listener)

    def __generateUpdates(self):
        while not self.__stopped:
            for object in self.__objectsList:
                self.__generateObjectUpdate(object)
            time.sleep(self.__generationInterval)

    def __generateObjectUpdate(self, object):
        self.__logInformation("Moving object: {} to position: {} {}".format(object.objectId, object.x, object.y))
        objectWidth = 2
        objectHeight = 2
        objectToUpdate = VisObject(VisObjectData('dummy', object.objectId, object.x, object.y, 0, objectWidth, objectHeight, object.properties()))
        objectToUpdate.updateAlerts(object.alerts())
        self.__abstractView.renderObject(objectToUpdate)
        self.__abstractView.updateAlerts(objectToUpdate)
        if object.x >= self.__modelMap.width():
            object.y += self.__modelMap.height() / 10
            object.x = self.__modelMap.x()
        else:
            object.x += self.__modelMap.width() / 20

    def __logError(self, message):
        for listener in self.__errorsListeners:
            listener.logError(message)

    def __logInformation(self, message):
        for listener in self.__errorsListeners:
            listener.logInformation(message)
