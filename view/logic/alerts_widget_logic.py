class AlertsWidgetLogic:

    def __init__(self):
        self.alerts = dict()
        self.alertsWidgetAccess = None

    def setWidgetAccess(self, widgetAccess):
        self.alertsWidgetAccess = widgetAccess

    def objectChanged(self, object):
        self.alerts[object.name()] = object.alerts()
        if self.alertsWidgetAccess is not None:
            self.alertsWidgetAccess.updateAlerts(object.name())

    def alertsForObject(self, objectName):
        try:
            return self.alerts[objectName]
        except KeyError:
            return {}

    def objectsNames(self):
        return self.alerts.keys()
