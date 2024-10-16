import sys
from business_rules.composition_root import CompositionRoot, DataInterfaces
from data_access.opc_data_access import OpcClientFactory
from data_access.fake_opc_data_access import FakeOpcClientFactory
from storage.configuration_storage import ConfigurationInJson
from storage.filesystem import Filesystem
from PyQt6.QtWidgets import QApplication
from view.mainframe import Mainframe

fakeOpcDataAccess = FakeOpcClientFactory()
opcDataAccess = OpcClientFactory()
configurationDataAccess = ConfigurationInJson(Filesystem())
dataInterfaces = DataInterfaces(opcDataAccess=opcDataAccess,
                                configurationDataAccess=configurationDataAccess,
                                fakeOpcDataAccess=fakeOpcDataAccess
                                )
businessRulesRoot = CompositionRoot(dataInterfaces=dataInterfaces)

app = QApplication([])

mainframe = Mainframe(businessRulesRoot)
mainframe.show()
sys.exit(app.exec())