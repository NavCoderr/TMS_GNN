from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt
from PyQt6.QtCore import *
from view.logic.output_widget_logic import LogLevel


class OutputWidgetAccess(QObject):
    def __init__(self, parent, outputWidget):
        super().__init__(parent=parent)
        self.outputWidget = outputWidget
        self.__outputWidgetDestroyed = False
        self.outputWidget.destroyed.connect(self.__onDestroyed)

    def updateLogs(self):
        if not self.__outputWidgetDestroyed:
            self.outputWidget.updateLogs.emit()

    def __onDestroyed(self):
        self.__outputWidgetDestroyed = True


class OutputWidget(QWidget):
    updateLogs = pyqtSignal(name="updateLogs")

    def __init__(self, parent, widgetLogic, logLevel):
        super().__init__(parent=parent)
        layout = QHBoxLayout()
        self.__logLevel = logLevel
        self.__logsWidget = QTreeWidget(parent=self)
        self.__logLevelSelectionWidget = QComboBox(parent=self)
        self.__clearLogsButton = QPushButton(parent=self)
        self.__logic = widgetLogic
        self.__setupClearButton()
        self.__setupLogLevelSelectionWidget()
        self.__applyLogLevel(logLevel)
        layout.addWidget(self.__logLevelSelectionWidget)
        layout.addWidget(self.__clearLogsButton)
        layout.addStretch()

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addWidget(self.__logsWidget)
        self.setLayout(mainLayout)
        self.updateLogs.connect(self.onLogsChanged)
        self.__outputWidgetAccess = OutputWidgetAccess(self, self)
        self.__logic.setViewAccess(self.outputWidgetAccess())

    def outputWidgetAccess(self):
        return self.__outputWidgetAccess

    def onLogsChanged(self):
        logs = self.__logic.getLogs(self.__logLevel)
        for logLevelName in logs:
            logsList = logs[logLevelName]
            self.__buildLogsItems(logLevelName, logsList)

    def __setupClearButton(self):
        self.__clearLogsButton.setText("Clear logs")
        self.__clearLogsButton.clicked.connect(self.__logic.clear)

    def __setupLogLevelSelectionWidget(self):
        self.__logLevelSelectionWidget.addItem("Errors", LogLevel.Error)
        self.__logLevelSelectionWidget.addItem("Warnings", LogLevel.Warning)
        self.__logLevelSelectionWidget.addItem("Information", LogLevel.Information)
        self.__logLevelSelectionWidget.addItem("Debug", LogLevel.Debug)
        self.__logLevelSelectionWidget.currentIndexChanged.connect(self.__onSelectionChanged)

    def __onSelectionChanged(self, index):
        logLevel = self.__logLevelSelectionWidget.itemData(index, Qt.ItemDataRole.UserRole)
        self.__applyLogLevel(logLevel)

    def __applyLogLevel(self, logLevel):
        self.__logLevel = logLevel
        self.__clearLogsWidget()
        self.__buildLogsWidget(logLevel)

    def __clearLogsWidget(self):
        self.__logsWidget.clear()

    def __buildLogsWidget(self, logLevel):
        self.__logsWidget.headerItem().setHidden(True)
        self.__logsWidget.setColumnCount(1)
        logs = self.__logic.getLogs(logLevel)
        for logLevelName in logs:
            logsList = logs[logLevelName]
            topLevelItem = QTreeWidgetItem()
            topLevelItem.setText(0, logLevelName)
            self.__logsWidget.addTopLevelItem(topLevelItem)
            self.__buildLogsItems(logLevelName, logsList)

    def __buildLogsItems(self, parentName, logsList):
        for index in range(0, self.__logsWidget.topLevelItemCount()):
            if self.__logsWidget.topLevelItem(index).text(0) == parentName:
                self.__logsWidget.topLevelItem(index).takeChildren()
                for logEntry in logsList:
                    logItem = QTreeWidgetItem()
                    logItem.setText(0, logEntry)
                    self.__logsWidget.topLevelItem(index).addChild(logItem)


class OutputDockWidget(QDockWidget):
    def __init__(self, parent, outputWidgetLogic):
        super().__init__(parent=parent)
        self.outputWidget = OutputWidget(self, outputWidgetLogic, LogLevel.Error)
        self.setWidget(self.outputWidget)
        self.setWindowTitle("Output")
