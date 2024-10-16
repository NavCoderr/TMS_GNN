class ObjectIdsGenerator:
    def __init__(self):
        self.objectsCount = 0

    def generateId(self):
        res = self.objectsCount
        self.objectsCount += 1
        return res


class VisObjectsRegistry:
    def __init__(self):
        self.__visObjectsById = {}
        self.__objectsIdsGenerator = ObjectIdsGenerator()

    def registerObject(self, visObject):
        self.__visObjectsById[visObject.getObjectId()] = visObject

    def unregisterObject(self, visObjectId):
        if self.object(visObjectId):
            del self.__visObjectsById[visObjectId]

    def object(self, objectId):
        try:
            return self.__visObjectsById[objectId]
        except KeyError:
            return None

    def idsGenerator(self):
        return self.__objectsIdsGenerator


