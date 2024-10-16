from business_rules.composition_root import CompositionRoot, DataInterfaces
from data_access.fake_opc_data_access import FakeOpcClientFactory
from storage.configuration_storage import ConfigurationInJson
from view.fakes.fake_ui.fake_mainframe import Mainframe


class FakeApp:
    def __init__(self):
        self.fakeOpcDataAccess = FakeOpcClientFactory()
        self.configurationDataAccess = ConfigurationInJson()
        dataInterfaces = DataInterfaces(opcDataAccess=None,
                                        configurationDataAccess=self.configurationDataAccess,
                                        fakeOpcDataAccess=self.fakeOpcDataAccess
                                        )
        self.businessRulesRoot = CompositionRoot(dataInterfaces=dataInterfaces)
        self.mainframe = Mainframe(self.businessRulesRoot)

