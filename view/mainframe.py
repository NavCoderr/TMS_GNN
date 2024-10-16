from view.widgets.map_widget import *
from view.logic.mainframe_logic import MainframeLogic
from view.widgets.map_dock_widget import MapDockWidget
from view.widgets.properties_pane import PropertiesDockWidget
from view.widgets.alerts_pane import AlertsDockWidget
from view.widgets.output_pane import OutputDockWidget
from view.widgets.configuration_picker import ConfigurationPickerDialog
from view.widgets.config_error_widget import IncorrectConfigWrapper
from view.widgets.configuration_wizard import ConfigurationWizard
from PyQt6.QtCore import QCoreApplication


class Mainframe(QMainWindow):
    def __init__(self, businessRules):
        super().__init__(parent=None)
        self.mainframeLogic = MainframeLogic(businessRules)

        self.setWindowTitle("Visualization")
        self.incorrectConfigWrapper = IncorrectConfigWrapper()
        self.outputDockWidget = OutputDockWidget(parent=self, outputWidgetLogic=self.mainframeLogic.outputLogic)
        self.alertsDockWidget = AlertsDockWidget(parent=self, logic=self.mainframeLogic.alerts)
        self.mapDockWidget = MapDockWidget(parent=self, startAppCallback=self.__start, mapWidgetLogic=self.mainframeLogic.mapWidgetLogic)
        self.propertiesDockWidget = PropertiesDockWidget(parent=self, propertiesLogic=self.mainframeLogic.propertiesLogic)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.propertiesDockWidget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.alertsDockWidget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.outputDockWidget)
        self.setCentralWidget(self.mapDockWidget)
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self.configurationPicker = ConfigurationPickerDialog(parent=self)
        self.configurationWizard = ConfigurationWizard(parent=self)
        self.mainframeLogic.userViewAdapter.setConfigurationPicker(self.configurationPicker)
        self.mainframeLogic.userViewAdapter.setIncorrectConfigDialog(self.incorrectConfigWrapper)
        self.mainframeLogic.userViewAdapter.setConfigurationWizard(self.configurationWizard)
        QCoreApplication.instance().aboutToQuit.connect(self.__stop)

        res = self.__initialize()
        if not res:
            self.mapDockWidget.mapPane.startVisualizationButton.setEnabled(False)

    def showMaximized(self) -> None:
        super().showMaximized()

    def __initialize(self):
        return self.mainframeLogic.initialize()

    def __start(self):
        self.mapDockWidget.mapPane.startVisualizationButton.setEnabled(False)
        self.mainframeLogic.start()

    def __stop(self):
        self.mainframeLogic.stop()


