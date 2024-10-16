import threading, time
from model.abstract_event_source import AbstractEventSource
from model.abstract_event_handler import AbstractEventHandler
from model.events import *


class FakeEventsSource(AbstractEventSource):
    def __init__(self):
        super().__init__()
        self.handlers = dict()
        self.eventsToSend = list()
        self.threadWorker = None

    def addHandler(self, handler):
        if issubclass(type(handler), AbstractEventHandler):
            self.handlers[id(handler)] = handler
        else:
            raise Exception("Not a handler!")

    def removeHandler(self, handler):
        if id(handler) not in self.handlers:
            raise Exception("Unregistering non-existing handler!")
        else:
            self.handlers.pop(id(handler))

    def registerAgvObject(self, objectId, name):
        event = RegisterObjectEvent(objectId=objectId, type="AGV", properties={"battery": "10%"}, width=4, height=3, name=name, frontLidarRange=1, rearLidarRange=1)
        self.__enqueueEvent(event)

    def updateObjectPosition(self, objectId, x, y):
        event = UpdateObjectPositionEvent(objectId=objectId, x=x, y=y)
        self.__enqueueEvent(event)

    def updateObjectRotation(self, objectId, rotation):
        event = UpdateObjectRotationEvent(objectId=objectId, rotation=rotation)
        self.__enqueueEvent(event)

    def updateObjectProperties(self, objectId, properties):
        event = UpdateObjectPropertiesEvent(objectId=objectId, properties=properties)
        self.__enqueueEvent(event)

    def updateObjectAlerts(self, objectId, alerts):
        event = UpdateObjectAlertsEvent(objectId=objectId, alerts=alerts)
        self.__enqueueEvent(event)

    def unregisterObject(self, objectId):
        self.__enqueueEvent(UnregisterObjectEvent(objectId=objectId))

    def refreshObject(self, objectId):
        self.__enqueueEvent(RefreshObjectEvent(objectId=objectId))

    def processEventsQueue(self):
        self.threadWorker = threading.Thread(target=self.__sendEvents)
        self.threadWorker.start()

    def __sendEvents(self):
        while len(self.eventsToSend) > 0:
            time.sleep(0.1)
            self.__broadcastEvent(self.eventsToSend.pop(0))
        self.__broadcastEvent(ShutdownEvent(unused=''))

    def __broadcastEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onEvent(event)

    def __enqueueEvent(self, event):
        self.eventsToSend.append(event)