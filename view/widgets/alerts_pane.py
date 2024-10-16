from PyQt6.QtWidgets import *
from PyQt6.QtCore import *


class AlertsWidgetAccess(QObject):
    def __init__(self, parent, alertsWidget):
        super().__init__(parent=parent)
        self.alertsWidget = alertsWidget

    def updateAlerts(self, objectName):
        self.alertsWidget.updateAlertsSignal.emit(objectName)


class AlertsWidget(QWidget):
    updateAlertsSignal = pyqtSignal(str, name="updateAlertsSignal")

    def __init__(self, parent, logic):
        super().__init__(parent=parent)
        self.__alertsAccess = AlertsWidgetAccess(self, self)
        self.__logic = logic
        self.__logic.setWidgetAccess(self.__alertsAccess)
        self.__alertsWidget = QTreeWidget(parent=self)
        self.updateAlertsSignal.connect(self.onAlertsChanged)
        self.__buildAlertsWidget()
        layout = QHBoxLayout()
        layout.addWidget(self.__alertsWidget)
        self.setLayout(layout)

    def onAlertsChanged(self, objectName):
        self.__buildOrUpdateAlertsForObject(objectName)

    def __buildOrUpdateAlertsForObject(self, objectName):
        alerts = self.__logic.alertsForObject(objectName)
        if alerts is not None:
            root = self.__getOrCreateAlertsRootForObject(objectName)
            for sectionName in alerts:
                self.__buildOrUpdateSection(sectionName, alerts[sectionName], root)

    def __buildAlertsWidget(self):
        self.__alertsWidget.headerItem().setText(0, "Alert Signal")
        self.__alertsWidget.headerItem().setText(1, "Alert Value")
        self.__alertsWidget.setColumnCount(2)
        objectNames = self.__logic.objectsNames()
        for objectName in objectNames:
            self.__buildOrUpdateAlertsForObject(objectName)

    def __getOrCreateAlertsRootForObject(self, objectName):
        root = None
        for i in range(0, self.__alertsWidget.topLevelItemCount()):
            if self.__alertsWidget.topLevelItem(i).text(0) == objectName:
                root = self.__alertsWidget.topLevelItem(i)
                break
        if root is None:
            root = QTreeWidgetItem()
            root.setText(0, objectName)
            self.__alertsWidget.addTopLevelItem(root)
        return root

    def __buildOrUpdateSection(self, sectionName, content, root):
        sectionRoot = None
        for i in range(0, root.childCount()):
            if root.child(i).text(0) == sectionName:
                sectionRoot = root.child(i)
        if sectionRoot is None:
            sectionRoot = QTreeWidgetItem()
            sectionRoot.setText(0, sectionName)
            root.addChild(sectionRoot)
        for alertSignalName in content:
            self.__buildOrUpdateAlertEntry(alertSignalName, content[alertSignalName], sectionRoot)

    def __buildOrUpdateAlertEntry(self, alertSignalName, alertSignalValue, sectionRoot):
        entryItem = None
        for i in range(0, sectionRoot.childCount()):
            if sectionRoot.child(i).text(0) == alertSignalName:
                entryItem = sectionRoot.child(i)
        if entryItem is None:
            entryItem = QTreeWidgetItem()
            entryItem.setText(0, alertSignalName)
            sectionRoot.addChild(entryItem)
        entryItem.setText(1, str(alertSignalValue))



class AlertsDockWidget(QDockWidget):
    def __init__(self, parent, logic):
        super().__init__(parent=parent)
        self.alertsWidget = AlertsWidget(self, logic)
        self.setWidget(self.alertsWidget)
        self.setWindowTitle("Alerts")
