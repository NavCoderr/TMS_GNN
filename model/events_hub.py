import time, sys, threading
from model.abstract_event_source import AbstractEventSource
from model.abstract_event_handler import AbstractEventHandler
from model.events import *


class EventsHub(AbstractEventSource, AbstractEventHandler):

    def __init__(self):
        super().__init__()
        self.sources = list()
        self.handlers = dict()
        self.handlersMap = dict()
        self.handlersMap[RegisterObjectEvent] = self.__onRegisterObjectEvent
        self.handlersMap[UpdateObjectPositionEvent] = self.__onUpdateObjectPositionEvent
        self.handlersMap[UpdateObjectRotationEvent] = self.__onUpdateObjectRotationEvent
        self.handlersMap[UpdateObjectPropertiesEvent] = self.__onUpdateObjectPropertiesEvent
        self.handlersMap[UpdateObjectAlertsEvent] = self.__onUpdateObjectAlertsEvent
        self.handlersMap[UnregisterObjectEvent] = self.__onUnregisterObjectEvent
        self.handlersMap[RefreshObjectEvent] = self.__onRefreshObjectEvent
        self.handlersMap[ShutdownEvent] = self.__onShutdownEvent
        self.eventsQueue = list()
        self.exit = False
        self.thread = threading.Thread(target=self.eventLoop)
        self.thread.daemon = True
        self.__event = threading.Event()

    def __del__(self):
        for source in self.sources:
            source.removeHandler(self)

    def addEventsSource(self, newSource):
        self.sources.append(newSource)
        newSource.addHandler(self)

    def addHandler(self, handler):
        self.handlers[id(handler)] = handler

    def removeHandler(self, handler):
        self.handlers.pop(id(handler))

    def onEvent(self, event):
        self.eventsQueue.append(event)
        self.__event.set()

    def eventLoop(self):
        while not self.exit:
            try:
                self.__event.wait()
                while len(self.eventsQueue) > 0:
                    event = self.eventsQueue.pop(0)
                    self.handlersMap[type(event)](event)
                time.sleep(0)
                self.__event.clear()
            except KeyboardInterrupt:
                sys.exit(0)

    def start(self):
        self.thread.start()

    def stop(self):
        self.__onShutdownEvent(None)

    def __onRegisterObjectEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onRegisterObject(event)

    def __onUpdateObjectPositionEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onUpdateObjectPosition(event)

    def __onUpdateObjectRotationEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onUpdateObjectRotation(event)

    def __onUpdateObjectPropertiesEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onUpdateObjectProperties(event)

    def __onUpdateObjectAlertsEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onUpdateObjectAlerts(event)

    def __onUnregisterObjectEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onUnregisterObject(event)

    def __onRefreshObjectEvent(self, event):
        for handlerId in self.handlers:
            self.handlers[handlerId].onRefreshObject(event)

    def __onShutdownEvent(self, event):
        self.exit = True
