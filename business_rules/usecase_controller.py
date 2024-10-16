import model.composition_root as model_root
import opc_adapter.composition_root as opc_root


class UseCaseController:
    def __init__(self, objectIdsGenerator, persistency):
        self.__objectFactoriesByType = dict()
        self.__objectIdsGenerator = objectIdsGenerator
        self.__view = None
        self.__persistency = persistency
        self.__errorSink = None
        self.__model = None

    def setView(self, view):
        self.__view = view

    def driveInitialization(self, modelRoot, opcRoot):
        self.__model = modelRoot
        self.__errorSink = modelRoot.errorSink()
        res = self.__view.askForConfigPath()
        configPath = res[0]
        editRequested = res[1]

        if not self.__persistency.fileExists(configPath):
            self.__persistency.setFilename(configPath)
            self.__view.driveConfigCreation(self.__persistency)
        # try:
        self.__persistency.read(configPath)
        if editRequested:
            self.__view.driveConfigEdit(self.__persistency)
        mapData = self.__getMapData()
        modelInitData = model_root.InitData(
            model_root.MapData(url=mapData[0], x=mapData[1], y=mapData[2], width=mapData[3], height=mapData[4]))

        if not modelRoot.initialize(modelInitData):
            return False
        if not opcRoot.initialize():
            return False

        self.__registerObjectsFromPersistency()
        return True

        # except:
        #     self.__view.onIncorrectConfig(configPath)
        #     return False

    def addObjectFactory(self, typeName, factory):
        self.__objectFactoriesByType[typeName] = factory

    def disconnectObject(self, visobjectId):
        self.__model.disconnectObject(visobjectId)

    def refreshObject(self, visobjectId):
        self.__model.refreshObject(visobjectId)

    def __getMapData(self):
        return self.__persistency.mapData()

    def __registerObject(self):
        registerData = self.__view.requestObjectRegistration()
        objectId = self.__objectIdsGenerator.generateId()
        self.__objectFactoriesByType[registerData['sourceType']].createObject(objectId, registerData, self.__errorSink).registerObject()

    def __registerObjectsFromPersistency(self):
        registerDataList = self.__persistency.objectsList()
        for registerData in registerDataList:
            objectId = self.__objectIdsGenerator.generateId()
            visObject = self.__objectFactoriesByType[registerData['sourceType']].createObject(objectId,
                                                                                  registerData,
                                                                                  self.__errorSink)
            if visObject:
                visObject.registerObject()
            else:
                self.__errorSink.logError("Object not created: " + registerData['name'])

