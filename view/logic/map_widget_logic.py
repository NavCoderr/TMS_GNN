from collections import namedtuple
from model.model_view import AbstractModelView
import math

Point = namedtuple('Point', 'x y')


class VisualizationWidgetLogic:
    def __init__(self, selection, usecaseController):
        self.__id = None
        self.__x = None
        self.__y = None
        self.__width = None
        self.__height = None
        self.__properties = None
        self.__selection = selection
        self.__usecaseController = usecaseController
        self.__objectObservers = dict()
        self.__modelX = None
        self.__modelY = None
        self.__name = "Dummy"
        self.__alerts = None
        self.__heading = None
        self.__frontLidarRadius = None
        self.__rearLidarRadius = None

    def initialize(self, id, x, y, width, height, properties, name, frontLidarRadius, rearLidarRadius):
        self.__id = id
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height
        self.__properties = properties
        self.__name = name
        self.__frontLidarRadius = frontLidarRadius
        self.__rearLidarRadius = rearLidarRadius

    def addObserver(self, observer):
        self.__objectObservers[id(observer)] = observer

    def removeObserver(self, observer):
        del self.__objectObservers[id(observer)]

    def setPosition(self, x, y):
        self.__x = x
        self.__y = y

    def setModelPosition(self, x, y):
        self.__modelX = x
        self.__modelY = y

    def setProperties(self, properties):
        self.__properties = properties

    def setAlerts(self, alerts):
        self.__alerts = alerts

    def setHeading(self, heading):
        self.__heading = heading

    def getBorderPoints(self):
        points = list()
        shape_width = self.__width
        shape_height = self.__height
        origin_x = self.__x
        origin_y = self.__y
        p1 = self.__rotatePoint(Point(int(-shape_width /2), int(-shape_height / 2)))
        p2 = self.__rotatePoint(Point(int(shape_width /2), int(-shape_height / 2)))
        p3 = self.__rotatePoint(Point(int(shape_width / 2), int(shape_height / 2)))
        p4 = self.__rotatePoint(Point(int(-shape_width / 2), int(shape_height / 2)))
        points.append(Point(x=int(p1.x + origin_x), y=int(p1.y + origin_y)))
        points.append(Point(x=int(p2.x + origin_x), y=int(p2.y + origin_y)))
        points.append(Point(x=int(p3.x + origin_x), y=int(p3.y + origin_y)))
        points.append(Point(x=int(p4.x + origin_x), y=int(p4.y + origin_y)))
        return points

    def getBoundingRect(self):
        top_left_x = self.__x - int(self.__width / 2)
        top_left_y = self.__y - int(self.__height / 2)
        return int(top_left_x), int(top_left_y), int(self.__width), int(self.__height)

    def getShapePoints(self):
        points = list()
        shape_width = self.__width
        shape_height = self.__height
        origin_x = self.__x
        origin_y = self.__y
        p1 = self.__rotatePoint(Point(0, int(-shape_height / 2)))
        p2 = self.__rotatePoint(Point(int(shape_width / 2), int(shape_height / 2)))
        p3 = self.__rotatePoint(Point(int(-shape_width / 2), int(shape_height / 2)))
        points.append(Point(x=int(p1.x + origin_x), y=int(p1.y + origin_y)))
        points.append(Point(x=int(p2.x + origin_x), y=int(p2.y + origin_y)))
        points.append(Point(x=int(p3.x + origin_x), y=int(p3.y + origin_y)))
        return points

    def __getEllipseBoundingRectAtPoint(self, radius, point):
        top_left_x = point.x - int(radius)
        top_left_y = point.y - int(radius)
        return int(top_left_x), int(top_left_y), int(radius*2), int(radius*2)

    def getFrontLidarEllipseRect(self):
        frontLidarPoint = self.getShapePoints()[0]
        return self.__getEllipseBoundingRectAtPoint(self.__frontLidarRadius, frontLidarPoint)

    def getRearLidarEllipseRect(self):
        shapePoints = self.getShapePoints()
        rearLidarPoint = Point(
            x = (shapePoints[1].x + shapePoints[2].x) / 2,
            y = (shapePoints[1].y + shapePoints[2].y) / 2)
        return self.__getEllipseBoundingRectAtPoint(self.__rearLidarRadius, rearLidarPoint)

    def properties(self):
        properties_dict = dict()
        properties_dict['x'] = str(self.__modelX)
        properties_dict['y'] = str(self.__modelY)
        properties_dict['heading'] = str(self.__heading)
        return { **self.__properties, **properties_dict }

    def updateSelection(self):
        self.__selection.updateSelection(self)

    def isSelected(self):
        return self.__selection.selectedObject() == self

    def broadcastObjectChanged(self):
        for observerId in self.__objectObservers:
            self.__objectObservers[observerId].objectChanged(self)

    def name(self):
        return self.__name

    def alerts(self):
        return self.__alerts

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def id(self):
        return self.__id

    def __rotatePoint(self, point):
        angle = self.__heading * -1
        sin = math.sin(math.radians(angle))
        cos = math.cos(math.radians(angle))
        x = point.x
        y = point.y
        newX = int(x * cos + y * sin)
        newY = int(y * cos - x * sin)
        return Point(x=newX, y=newY)

    def disconnect(self):
        self.__usecaseController.disconnectObject(self.__id)

    def reconnect(self):
        self.__usecaseController.refreshObject(self.__id)


class MapWidgetLogic:
    def __init__(self, selection, alerts, usecaseController):
        self.modelMap = None
        self.viewAccess = None
        self.usecaseController = usecaseController
        self.objectsDict = {}
        self.xScaling = 1.0
        self.yScaling = 1.0
        self.mapOrigin = 0, 0
        self.rowHeight = 5
        self.columnWidth = 5
        self.selection = selection
        self.alerts = alerts

    def setViewAccess(self, viewAccess):
        self.viewAccess = viewAccess

    def updateObject(self, visobject):
        if visobject.getObjectId() not in self.objectsDict:
            self.registerObject(visobject)

        objectToUpdate = self.objectsDict[visobject.getObjectId()]
        x = self.offsetX(visobject.getX())
        y = self.offsetY(visobject.getY())

        viewSize = self.viewAccess.size()
        if x < 0 or x > viewSize.width():
            return
        if y < 0 or y > viewSize.height():
            return

        objectToUpdate.setPosition(x, y)
        objectToUpdate.setModelPosition(visobject.getX(), visobject.getY())
        objectToUpdate.setHeading(visobject.getRotation())
        objectToUpdate.setProperties(visobject.getProperties())
        objectToUpdate.broadcastObjectChanged()
        self.viewAccess.updateView()

    def updateObjectProperties(self, visobject):
        try:
            objectToUpdate = self.objectsDict[visobject.getObjectId()]
            objectToUpdate.setProperties(visobject.getProperties())
            objectToUpdate.broadcastObjectChanged()
            self.viewAccess.updateView()
        except KeyError:
            pass

    def updateObjectAlerts(self, visobject):
        try:
            objectToUpdate = self.objectsDict[visobject.getObjectId()]
            objectToUpdate.setAlerts(visobject.getAlerts())
            objectToUpdate.broadcastObjectChanged()
            self.viewAccess.updateView()
        except KeyError:
            pass

    def registerObject(self, visobject):
        self.objectsDict[visobject.getObjectId()] = \
            VisualizationWidgetLogic(self.selection, self.usecaseController)
        self.objectsDict[visobject.getObjectId()].initialize(
            id=visobject.getObjectId(),
            x=self.offsetX(visobject.getX()),
            y=self.offsetY(visobject.getY()),
            width=visobject.getWidth() * self.xScaling,
            height=visobject.getHeight() * self.yScaling,
            properties=visobject.getProperties(),
            name = visobject.getName(),
            frontLidarRadius = visobject.getFrontLidarRange() * self.xScaling,
            rearLidarRadius = visobject.getRearLidarRange() * self.xScaling
        )
        self.objectsDict[visobject.getObjectId()].addObserver(self.alerts)
        self.viewAccess.addObject(self.objectsDict[visobject.getObjectId()])

    def unregisterObject(self, visobjectId):
        try:
            self.objectsDict[visobjectId].removeObserver(self.alerts)
            del self.objectsDict[visobjectId]
            self.viewAccess.eraseObject(visobjectId)
            self.viewAccess.updateView()
        except KeyError:
            pass

    def renderMap(self, modelMap):
        self.modelMap = modelMap
        self.viewAccess.setPixmapUrl(self.url())
        self.updateScaling()
        self.updateOrigin()
        self.viewAccess.updateGrid(self.columnWidth * self.xScaling, self.rowHeight * self.yScaling)
        self.viewAccess.showMap()

    def updateScaling(self):
        viewSize = self.viewAccess.size()
        modelSize = self.modelMap.size()
        self.xScaling = viewSize.width() / modelSize[0]
        self.yScaling = viewSize.height() / modelSize[1]

    def updateOrigin(self):
        self.mapOrigin = self.modelMap.x(), self.modelMap.y()

    def offsetX(self, x):
        modelDistance = x - self.mapOrigin[0]
        return modelDistance * self.xScaling

    def offsetY(self, y):
        modelDistance = y - self.mapOrigin[1]
        scaledDistance = modelDistance * self.yScaling
        return self.viewAccess.size().height() - scaledDistance

    def url(self):
        return self.modelMap.url()


class ModelViewToMapLogicAdapter(AbstractModelView):
    def __init__(self, mapLogic: MapWidgetLogic):
        super().__init__()
        self.mapLogic = mapLogic

    def renderObject(self, visObject):
        self.mapLogic.updateObject(visObject)

    def updateProperties(self, visObject):
        self.mapLogic.updateObjectProperties(visObject)

    def updateAlerts(self, visObject):
        self.mapLogic.updateObjectAlerts(visObject)

    def cleanupObject(self, visObjectId):
        self.mapLogic.unregisterObject(visObjectId)

    def showCollision(self, collidingObjects):
        pass

    def renderMap(self, visMap):
        self.mapLogic.renderMap(visMap)
