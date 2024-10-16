class Map:
    def __init__(self, objectsRegistry):
        self.__x = None
        self.__y = None
        self.__width = None
        self.__height = None
        self.__url = None
        self.view = None
        self.objectsRegistry = objectsRegistry
        self.collidedObjects = list()

    def initialize(self, x, y, width, height, view, url):
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height
        self.__url = url
        self.view = view
        self.view.renderMap(self)

    def onChangedObjects(self, changedObjects):
        for changedObject in changedObjects:
            self.__onObjectChanged(changedObject)

    def onObjectsPropertiesChanged(self, changedObjects):
        for changedObject in changedObjects:
            self.__onObjectPropertiesChanged(changedObject)

    def onObjectsAlertsChanged(self, changedObjects):
        for changedObject in changedObjects:
            self.__onObjectAlertsChanged(changedObject)

    def topLeft(self):
        return self.__x, self.__y

    def bottomRight(self):
        return self.__x + self.__width, self.__y + self.__height

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def width(self):
        return self.__width

    def height(self):
        return self.__height

    def size(self):
        return self.width(), self.height()

    def url(self):
        return self.__url

    def isValidPosition(self, x, y):
        if x >= self.topLeft()[0] and y >= self.topLeft()[1]:
            if x <= self.bottomRight()[0] and y <= self.bottomRight()[1]:
                return True
        return False

    def __onObjectChanged(self, visObjectId):
        visObject = self.objectsRegistry.object(visObjectId)
        if visObject:
            self.view.renderObject(visObject)
            self.__updateCollisions()
        else:
            self.view.cleanupObject(visObjectId)

    def __onObjectPropertiesChanged(self, visObjectId):
        visObject = self.objectsRegistry.object(visObjectId)
        if visObject:
            self.view.updateProperties(visObject)

    def __onObjectAlertsChanged(self, visObjectId):
        visObject = self.objectsRegistry.object(visObjectId)
        if visObject:
            self.view.updateAlerts(visObject)

    def __updateCollisions(self):
        if len(self.collidedObjects) > 0:
            self.view.showCollision(self.collidedObjects)
