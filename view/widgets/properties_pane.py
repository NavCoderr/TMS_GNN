from PyQt6.QtWidgets import *


class PropertiesDockWidget(QDockWidget):
    def __init__(self, parent, propertiesLogic):
        super().__init__(parent=parent)
        self.propertiesWidget = PropertiesWidget(parent = self, properties={}, widgetLogic=propertiesLogic)
        self.setWidget(self.propertiesWidget)
        self.setWindowTitle("Properties")


class PropertiesWidget(QWidget):
    
    def __init__(self, parent, properties, widgetLogic):
        super().__init__(parent=parent)
        layout = QHBoxLayout()

        self.propertiesGrid = QTableWidget(self)
        self.propertiesGrid.setColumnCount(2)
        self.propertiesGrid.setHorizontalHeaderItem(0, QTableWidgetItem("Property"))
        self.propertiesGrid.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))

        self.setProperties(properties)

        self.propertiesGrid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.propertiesGrid)
        self.setLayout(layout)
        self.__logic = widgetLogic
        self.__logic.setViewAccess(self)

    def setProperties(self, properties):
        self.propertiesGrid.setRowCount(len(properties))
        currentRow = 0
        for propertyName in properties:
            self.propertiesGrid.setItem(currentRow, 0, QTableWidgetItem(propertyName))
            self.propertiesGrid.setItem(currentRow, 1, QTableWidgetItem(str(properties[propertyName])))
            currentRow += 1
