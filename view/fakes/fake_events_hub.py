import random


class FakeObject:
    def __init__(self, objectId, x, y):
        self.objectId = objectId
        self.x = x
        self.y = y
        self.rotation = 0
        self.width = 2
        self.height = 2

    def properties(self):
        res = dict()
        res['id'] = self.objectId
        res['x'] = self.x
        res['y'] = self.y
        res['battery'] = random.randint(0, 100)
        return res

    def alerts(self):
        res = dict()
        res['Information'] = dict()
        res['Warnings'] = dict()
        res['Information']['Information1'] = "Information value 1"
        res['Warnings']['Warning1'] = "Warning value 1"
        return res


class FakeModelMap:
    def __init__(self, mapSize):
        self.__x = mapSize.x
        self.__y = mapSize.y
        self.__width = mapSize.width
        self.__height = mapSize.height

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


class FakeBusinessRules:
    def __init__(self, objectIds, mapSize, updatesGenerator):
        self.modelView = None
        self.userView = None
        self.updatesGenerator = updatesGenerator
        self.objectsIds = objectIds
        self.fakeModelMap = FakeModelMap(mapSize)
        self.errorsListener = None

    def setViewInterfaces(self, viewInterfaces):
        self.modelView = viewInterfaces.modelView
        self.userView = viewInterfaces.userView

    def initialize(self):
        self.updatesGenerator.initialize(self.modelView, self.objectsIds, self.fakeModelMap)
        self.updatesGenerator.addErrorListener(self.errorsListener)
        self.modelView.renderMap(self.fakeModelMap)

    def startApp(self):
        self.updatesGenerator.start()

    def addErrorsListener(self, listener):
        self.errorsListener = listener
