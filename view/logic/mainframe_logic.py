from view.logic.map_widget_logic import *
from view.logic.user_view import *
from view.logic.selection import *
from view.logic.properties_logic import *
from view.logic.output_widget_logic import *
from view.logic.alerts_widget_logic import *
from business_rules.composition_root import ViewInterfaces


class MainframeLogic:
    def __init__(self, businessRules):
        self.businessRules = businessRules
        self.outputLogic = OutputWidgetLogic()
        self.alerts = AlertsWidgetLogic()
        self.selection = Selection()
        self.mapWidgetLogic = MapWidgetLogic(self.selection,
                                             self.alerts,
                                             self.businessRules.useCaseController)
        self.propertiesLogic = PropertiesLogic(self.selection)
        self.userViewAdapter = QtViewToAbstractUserView()
        self.modelViewAdapter = ModelViewToMapLogicAdapter(self.mapWidgetLogic)

    def initialize(self):
        modelView = self.modelViewAdapter
        userView = self.userViewAdapter
        viewInterfaces = ViewInterfaces(modelView=modelView, userView=userView)

        self.businessRules.addErrorsListener(self.outputLogic)
        self.businessRules.setViewInterfaces(viewInterfaces)
        return self.businessRules.initialize()

    def start(self):
        self.businessRules.startApp()

    def stop(self):
        self.businessRules.killApp()