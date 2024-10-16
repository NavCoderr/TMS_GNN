class AbstractModelView:
    def __init__(self):
        pass

    def renderObject(self, visObject):
        raise Exception("Not implemented!")

    def updateProperties(self, visObject):
        raise Exception("Not implemented!")

    def updateAlerts(self, visObject):
        raise Exception("Not implemented!")

    def cleanupObject(self, visObjectId):
        raise Exception("Not implemented!")

    def showCollision(self, collidingObjects):
        raise Exception("Not implemented!")

    def renderMap(self, visMap):
        raise Exception("Not implemented!")

