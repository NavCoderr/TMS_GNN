class VisObjectData:
    def __init__(self, name, objectId, x, y, rotation, width, height, properties, frontLidarRange, rearLidarRange):
        self.name = name
        self.objectId = objectId
        self.x = x
        self.y = y
        self.rotation = rotation
        self.width = width
        self.height = height
        self.properties = properties
        self.frontLidarRange = frontLidarRange
        self.rearLidarRange = rearLidarRange


class VisObject:
    def __init__(self, visObjectData):
        self.__name = visObjectData.name
        self.__objectId = visObjectData.objectId
        self.__x = visObjectData.x
        self.__y = visObjectData.y
        self.__rotation = visObjectData.rotation
        self.__width = visObjectData.width
        self.__height = visObjectData.height
        self.__properties = visObjectData.properties
        self.__alerts = None
        self.__frontLidarRange = visObjectData.frontLidarRange
        self.__rearLidarRange = visObjectData.rearLidarRange

    def setX(self, x):
        self.__x = x

    def setY(self, y):
        self.__y = y

    def setPosition(self, x, y):
        self.setX(x)
        self.setY(y)

    def setRotation(self, rotation):
        self.__rotation = rotation

    def updateProperties(self, properties):
        self.__properties = properties

    def updateAlerts(self, alerts):
        self.__alerts = alerts

    def getObjectId(self):
        return self.__objectId

    def getX(self):
        return self.__x

    def getY(self):
        return self.__y

    def getWidth(self):
        return self.__width

    def getHeight(self):
        return self.__height

    def getRotation(self):
        return self.__rotation

    def getBoundingRect(self):
        topLeft = (self.__x, self.__y)
        bottomRight = (topLeft[0] + self.__width, topLeft[1] + self.__height)
        return topLeft, bottomRight

    def getProperties(self):
        return self.__properties

    def getAlerts(self):
        return self.__alerts

    def getName(self):
        return self.__name

    def getFrontLidarRange(self):
        return self.__frontLidarRange

    def getRearLidarRange(self):
        return self.__rearLidarRange


