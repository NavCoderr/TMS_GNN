from view.logic.mainframe_logic import MainframeLogic
from view.fakes.fake_ui.fake_alerts_widget import FakeAlertsWidget
from view.fakes.fake_ui.fake_map_widget import FakeMapWidget
from view.fakes.fake_ui.fake_output_widget import FakeOutputWidget
from view.fakes.fake_ui.fake_properties_widget import FakePropertiesWidget


class Mainframe:
    def __init__(self, businessRules):
        self.mainframeLogic = MainframeLogic(businessRules)

        self.outputWidget = FakeOutputWidget(widgetLogic=self.mainframeLogic.outputLogic)
        self.alertsWidget = FakeAlertsWidget(widgetLogic=self.mainframeLogic.alerts)
        self.mapWidget = FakeMapWidget(widgetLogic=self.mainframeLogic.mapWidgetLogic)
        self.propertiesWidget = FakePropertiesWidget(widgetLogic=self.mainframeLogic.propertiesLogic)

    def setViewSize(self, width, height):
        self.mapWidget.setSize((width, height))

    def initialize(self):
        self.mainframeLogic.initialize()

    def start(self):
        self.mainframeLogic.start()

    def stop(self):
        self.mainframeLogic.stop()


