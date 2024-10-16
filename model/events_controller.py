from model.entities.agv import *


class EventsController:
    def __init__(self, visobjectsRegistry, eventsSource, map, errorSink):
        self.visobjectsRegistry = visobjectsRegistry
        self.eventsSource = eventsSource
        self.eventsSource.addHandler(self)
        self.objectsObservers = list()
        self.map = map
        self.errorSink = errorSink
        self.registerObjectObserver(self.map)

    def registerObjectObserver(self, observer):
        self.objectsObservers.append(observer)

    def broadcastObjectsChanged(self, changedObjects):
        for observer in self.objectsObservers:
            observer.onChangedObjects(changedObjects)

    def broadcastObjectsPropertiesChanged(self, changedObjects):
        for observer in self.objectsObservers:
            observer.onObjectsPropertiesChanged(changedObjects)

    def broadcastObjectsAlertsChanged(self, changedObjects):
        for observer in self.objectsObservers:
            observer.onObjectsAlertsChanged(changedObjects)

    def __del__(self):
        self.eventsSource.removeHandler(self)

    def onRegisterObject(self, registerObjectEvent):
        newObjectId = registerObjectEvent.objectId
        newObjectName = registerObjectEvent.name
        if not self.visobjectsRegistry.object(newObjectId):
            if registerObjectEvent.type == 'AGV':
                visObjectData = VisObjectData(newObjectName,
                                              newObjectId,
                                              0, 0, 0,
                                              registerObjectEvent.width,
                                              registerObjectEvent.height,
                                              registerObjectEvent.properties,
                                              registerObjectEvent.frontLidarRange,
                                              registerObjectEvent.rearLidarRange)
                agvObjectData = AgvObjectData(visObjectData)
                self.visobjectsRegistry.registerObject(AgvObject(agvObjectData))
                self.broadcastObjectsChanged([newObjectId])
        else:
            self.errorSink.logError("Object {} already exists!".format(newObjectId))

    def onUpdateObjectPosition(self, updateObjectPositionEvent):
        objectId = updateObjectPositionEvent.objectId
        x = updateObjectPositionEvent.x
        y = updateObjectPositionEvent.y

        visObject = self.visobjectsRegistry.object(objectId)
        if visObject:
            if self.map.isValidPosition(x, y):
                visObject.setPosition(x, y)
                self.broadcastObjectsChanged([objectId])
            else:
                self.errorSink.logError("Invalid position: ({}, {})".format(x, y))
        else:
            self.errorSink.logError("Object {} does not exists!".format(objectId))

    def onUpdateObjectRotation(self, updateObjectRotationEvent):
        objectId = updateObjectRotationEvent.objectId
        value = updateObjectRotationEvent.rotation
        try:
            rotation = float(value) % 360

            visObject = self.visobjectsRegistry.object(objectId)
            if visObject:
                visObject.setRotation(rotation)
                self.broadcastObjectsChanged([objectId])
            else:
                self.errorSink.logError("Object {} does not exists!".format(objectId))
        except ValueError:
            self.errorSink.logError("Rotation value: '{}' is not a number!".format(value))

    def onUpdateObjectProperties(self, updateObjectPropertiesEvent):
        objectId = updateObjectPropertiesEvent.objectId
        properties = updateObjectPropertiesEvent.properties

        visObject = self.visobjectsRegistry.object(objectId)
        if visObject:
            visObject.updateProperties(properties)
            self.broadcastObjectsPropertiesChanged([objectId])
        else:
            self.errorSink.logError("Object {} does not exists!".format(objectId))

    def onUpdateObjectAlerts(self, updateObjectAlertsEvent):
        objectId = updateObjectAlertsEvent.objectId
        alerts = updateObjectAlertsEvent.alerts

        visObject = self.visobjectsRegistry.object(objectId)
        if visObject:
            visObject.updateAlerts(alerts)
            self.broadcastObjectsAlertsChanged([objectId])
        else:
            self.errorSink.logError("Object {} does not exists!".format(objectId))

    def onUnregisterObject(self, unregisterObjectEvent):
        objectId = unregisterObjectEvent.objectId
        self.visobjectsRegistry.unregisterObject(objectId)
        self.broadcastObjectsChanged([objectId])

    def onRefreshObject(self, refreshObjectEvent):
        objectId = refreshObjectEvent.objectId
        self.broadcastObjectsChanged([objectId])
