import model.composition_root as model_root
import opc_adapter.composition_root as opc_root
from business_rules.usecase_controller import UseCaseController
from collections import namedtuple

ViewInterfaces = namedtuple('ViewInterfaces', 'modelView userView')
DataInterfaces = namedtuple('DataInterfaces', 'opcDataAccess configurationDataAccess fakeOpcDataAccess')


class CompositionRoot:
    def __init__(self, dataInterfaces):
        self.modelRoot = model_root.CompositionRoot()

        self.opcRoot = opc_root.CompositionRoot(self.modelRoot.eventsHub(),
                                                dataInterfaces.opcDataAccess,
                                                dataInterfaces.fakeOpcDataAccess,
                                                self.modelRoot.errorSink())

        self.useCaseController = UseCaseController(self.modelRoot.objectsIdsGenerator(),
                                                   dataInterfaces.configurationDataAccess)

        self.useCaseController.addObjectFactory("OPC", self.opcRoot.objectsFactory())
        self.useCaseController.addObjectFactory("FAKE_OPC", self.opcRoot.fakesFactory())

    def setViewInterfaces(self, viewInterfaces):
        self.modelRoot.setView(viewInterfaces.modelView)
        self.useCaseController.setView(viewInterfaces.userView)

    def initialize(self):
        return self.useCaseController.driveInitialization(self.modelRoot, self.opcRoot)

    def startApp(self):
        self.modelRoot.startProcessingEvents()

    def killApp(self):
        self.modelRoot.stopProcessingEvents()

    def addErrorsListener(self, listener):
        self.modelRoot.addErrorListener(listener)