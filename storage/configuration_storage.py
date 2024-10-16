from model.composition_root import MapData
from os.path import exists
import json


class ConfigBuilder:
    def __init__(self):
        self.dictToDump = dict()

    def setMapData(self, mapData):
        self.dictToDump["mapData"] = mapData

    def setObjects(self, objects):
        self.dictToDump["objects"] = objects

    def reset(self):
        self.dictToDump = dict()


class ConfigurationInJson:
    def __init__(self, filesystem):
        super().__init__()
        self.__data = None
        self.__filename = ""
        self.__builder = ConfigBuilder()
        self.__fs = filesystem

    def fileExists(self, filename):
        return exists(filename)

    def setFilename(self, filename):
        self.__filename = filename

    def read(self, filename):
        self.__filename = filename
        self.__data = json.loads(self.__fs.readFile(self.__filename))

    def hasMapData(self):
        try:
            tmp = self.__data['mapData']
            return True
        except KeyError:
            return False

    def saveMapData(self, mapData):
        self.__builder = ConfigBuilder()
        toDump = dict()
        toDump['url'] = mapData[0]
        toDump['x'] = mapData[1]
        toDump['y'] = mapData[2]
        toDump['width'] = mapData[3]
        toDump['height'] = mapData[4]
        self.__builder.setMapData(toDump)

    def saveObjects(self, objects):
        objectsAsDicts = list()
        for __obj in objects:
            objectAsDict = __obj._asdict()
            properties = dict()
            for property in __obj.properties:
                properties[property[0]] = property[1]
            alerts = dict()
            for alert in __obj.alerts:
                alerts[alert[0]] = alert[1]
            objectAsDict['properties'] = properties
            objectAsDict['alerts'] = alerts
            objectsAsDicts.append(objectAsDict)

        self.__builder.setObjects(objectsAsDicts)

    def write(self):
        toWrite = json.dumps(self.__builder.dictToDump, indent=2)
        with open(self.__filename, 'w') as f:
            f.write(toWrite)

    def mapData(self):
        node = self.__data['mapData']
        mapData = MapData(x=node['x'], y=node['y'], width=node['width'], height=node['height'], url=node['url'])
        return mapData

    def objectsList(self):
        objectsList = list()
        objects = self.__data['objects']
        for object in objects:
            registerData = dict()
            registerData['name'] = object['name']
            registerData['sourceType'] = object['sourceType']
            registerData['width'] = float(object['width'])
            registerData['height'] = float(object['height'])
            registerData['frontLidarRange'] = float(object['frontLidarRange'])
            registerData['rearLidarRange'] = float(object['rearLidarRange'])
            registerData['type'] = "AGV"
            registerData['connectionString'] = object['connectionString']
            registerData['xSignal'] = object['xSignal'].split('/')
            registerData['ySignal'] = object['ySignal'].split('/')
            registerData['rotationSignal'] = object['headingSignal'].split('/')
            registerData['updateInterval'] = float(object['updateInterval'])
            registerData['properties'] = dict()
            for propertyNode in object['properties']:
                registerData['properties'][propertyNode] = object['properties'][propertyNode].split('/')
            registerData['alerts'] = dict()
            for alertNode in object['alerts']:
                registerData['alerts'][alertNode] = object['alerts'][alertNode].split('/')

            objectsList.append(registerData)
        return objectsList
