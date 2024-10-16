from model.model_view import AbstractModelView
from model.composition_root import MapData
from business_rules.abstract_user_view import AbstractUserView


class ConsoleModelView(AbstractModelView):
    def __init__(self):
        super().__init__()

    def renderObject(self, visObject):
        print("Rendering object: {} {} {}".format(visObject.getObjectId(), visObject.getX(), visObject.getY()))

    def cleanupObject(self, visObjectId):
        print("Cleaning up object: {}".format(visObjectId.getObjectId()))

    def showCollision(self, collidingObjects):
        raise Exception("Not implemented!")

    def renderMap(self, visMap):
        print("Rendering map: {} {} {} {}".format(visMap.topLeft()[0], visMap.topLeft()[1],
                                                  visMap.bottomRight()[0], visMap.bottomRight()[1]))


class ConsoleUserView(AbstractUserView):
    def __init__(self):
        super().__init__()

    def requestMapData(self):
        str = input("Provide map coordinates(topLeft.x topLeft.y bottomRight.x bottomRight.y: ")
        splitted = str.split(' ')
        return MapData(x=int(splitted[0]), y=int(splitted[1]), width=int(splitted[2]), height=int(splitted[3]))

    def askForObjectsRegistration(self):
        return False

    def requestObjectRegistration(self):
        str = input("Register AGV object(serverUrl,xSignal[path],ySignal[path],updateInterval[s]: ")
        splitted = str.split(',')

        registerData = dict()
        registerData['sourceType'] = "OPC"
        registerData['width'] = 10
        registerData['height'] = 15
        registerData['type'] = "AGV"
        registerData['connectionString'] = splitted[0]
        registerData['xSignal'] = splitted[1].split('/')
        registerData['ySignal'] = splitted[2].split('/')
        registerData['updateInterval'] = float(splitted[3])

        return registerData

