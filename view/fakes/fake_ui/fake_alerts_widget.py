class FakeAlertsWidget:
    def __init__(self, widgetLogic):
        widgetLogic.setWidgetAccess(self)

    def updateAlerts(self, objectName):
        pass

