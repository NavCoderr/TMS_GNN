from model.model_view import AbstractModelView
from copy import deepcopy


class FakeView(AbstractModelView):
    def __init__(self):
        super().__init__()
        self.renderedMap = None
        self.renderedObjects = list()
        self.knownObjects = dict()

    def renderObject(self, visObject):
        if visObject.getObjectId() not in self.knownObjects:
            self.knownObjects[visObject.getObjectId()] = visObject
        self.renderedObjects.append(deepcopy(visObject))

    def updateProperties(self, visObject):
        return self.renderObject(visObject)

    def updateAlerts(self, visObject):
        return self.renderObject(visObject)

    def cleanupObject(self, visObjectId):
        if visObjectId in self.knownObjects:
            self.knownObjects.pop(visObjectId)

    def showCollision(self, collidingObjects):
        pass

    def renderMap(self, visMap):
        self.renderedMap = visMap

    def lastRenderedObject(self):
        return self.renderedObjects[len(self.renderedObjects)-1]

