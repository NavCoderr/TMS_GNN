from PyQt6.QtWidgets import *
from collections import namedtuple
from view.logic.configuration_widgets_builder import ConfigurationWidgetsBuilder, ObjectData


class LabeledLineEdit(QWidget):
    def __init__(self, parent, label):
        super().__init__(parent=parent)
        self.label = QLabel(label)
        self.lineEdit = QLineEdit()
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        self.setLayout(layout)


class FilePickerWidget(QWidget):
    def __init__(self, parent, label, buttonLabel):
        super().__init__(parent=parent)
        self.label = QLabel(label)
        self.lineEdit = QLineEdit()
        self.button = QPushButton(buttonLabel)
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.button.clicked.connect(self.__showFileDialog)

    def __showFileDialog(self):
        file = QFileDialog.getOpenFileName(parent=self,
                                                  caption="Pick map file",
                                                  directory="",
                                                  filter="Images (*.png *.bmp *.jpg)")[0]
        self.lineEdit.setText(file)


class LabeledSpinBox(QWidget):
    def __init__(self, parent, label, min, max, decimals, step):
        super().__init__(parent=parent)
        self.label = QLabel(label)
        self.spinbox = QDoubleSpinBox(parent=self)
        self.spinbox.setMaximum(max)
        self.spinbox.setMinimum(min)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setSingleStep(step)
        layout = QHBoxLayout()
        layout.addStretch()
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)


class MapDataWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.groupBox = QGroupBox(parent=self, title="Map data")
        self.xSpinBox = LabeledSpinBox(self, "X: ", -1000, 1000, 5, 1)
        self.ySpinBox = LabeledSpinBox(self, "Y: ", -1000, 1000, 5, 1)
        self.widthSpinBox = LabeledSpinBox(self, "Width: ", 0, 1000, 5, 1)
        self.heightSpinBox = LabeledSpinBox(self, "Height: ", 0, 1000, 5, 1)
        self.filePicker = FilePickerWidget(self, "Map file: ", "Browse")
        groupboxLayout = QVBoxLayout()
        groupboxLayout.addWidget(self.filePicker)
        groupboxLayout.addWidget(self.xSpinBox)
        groupboxLayout.addWidget(self.ySpinBox)
        groupboxLayout.addWidget(self.widthSpinBox)
        groupboxLayout.addWidget(self.heightSpinBox)
        groupboxLayout.addStretch()
        self.groupBox.setLayout(groupboxLayout)
        layout = QVBoxLayout()
        layout.addWidget(self.groupBox)
        self.setLayout(layout)

    def mapData(self):
        mapUrl = self.filePicker.lineEdit.text()
        x = self.xSpinBox.spinbox.value()
        y = self.ySpinBox.spinbox.value()
        width = self.widthSpinBox.spinbox.value()
        height = self.heightSpinBox.spinbox.value()
        return mapUrl, x, y, width, height


class ManageObjectWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.pickObjectTypeLabel = QLabel("Object type: ")
        self.pickObjectTypeCombo = QComboBox(parent=self)
        self.pickObjectTypeCombo.addItem("OPC")
        self.pickObjectTypeCombo.addItem("FAKE_OPC")
        self.addObjectButton = QPushButton("Add object")
        self.editObjectButton = QPushButton("Edit object")
        self.copyObjectButton = QPushButton("Copy object")
        self.removeObjectButton = QPushButton("Remove object")
        layout = QHBoxLayout()
        layout.addWidget(self.pickObjectTypeLabel)
        layout.addWidget(self.pickObjectTypeCombo)
        layout.addWidget(self.addObjectButton)
        layout.addWidget(self.editObjectButton)
        layout.addWidget(self.copyObjectButton)
        layout.addWidget(self.removeObjectButton)
        layout.addStretch()
        self.setLayout(layout)
        self.copyObjectButton.setEnabled(False)
        self.editObjectButton.setEnabled(False)
        self.removeObjectButton.setEnabled(False)


class TwoColumnsGrid(QWidget):
    def __init__(self, parent, firstColumnName, secondColumnName, buttonLabel):
        super().__init__(parent=parent)
        self.grid = QTableWidget()
        self.grid.setColumnCount(2)
        labels = list()
        labels.append(firstColumnName)
        labels.append(secondColumnName)
        self.grid.setHorizontalHeaderLabels(labels)
        self.button = QPushButton(buttonLabel)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.button)
        buttonLayout.addStretch()
        layout = QVBoxLayout()
        layout.addWidget(self.grid)
        layout.addLayout(buttonLayout)
        self.setLayout(layout)
        self.button.clicked.connect(self.__addRow)
        self.grid.horizontalHeader().setStretchLastSection(True)

    def __addRow(self):
        rows = self.grid.rowCount()
        self.grid.setRowCount(rows+1)

    def dumpGrid(self):
        items = list()
        for i in range(0, self.grid.rowCount()):
            item = self.grid.item(i, 0).text(), self.grid.item(i, 1).text()
            items.append(item)
        return items

    def fill(self, rows):
        self.grid.setRowCount(len(rows))
        for i in range(0, len(rows)):
            item1 = QTableWidgetItem(rows[i][0])
            item2 = QTableWidgetItem(rows[i][1])
            self.grid.setItem(i, 0, item1)
            self.grid.setItem(i, 1, item2)



class AddObjectDialog(QDialog):
    def __init__(self, parent, sourceType, onSubmit):
        super().__init__(parent=parent)
        self.name = LabeledLineEdit(parent=self, label="Name: ")
        self.sourceType = sourceType
        self.connectionString = LabeledLineEdit(parent=self, label="Connection string: ")
        self.width = LabeledSpinBox(self, "Width: ", 0, 1000, 5, 1)
        self.height = LabeledSpinBox(self, "Height: ", 0, 1000, 5, 1)
        self.frontLidarRange = LabeledSpinBox(self, "Front lidar range: ", 0, 1000, 5, 1)
        self.rearLidarRange = LabeledSpinBox(self, "Rear lidar range: ", 0, 1000, 5, 1)
        self.xSignal = LabeledLineEdit(parent=self, label="X coordinate signal: ")
        self.ySignal = LabeledLineEdit(parent=self, label="Y coordinate signal: ")
        self.headingSignal = LabeledLineEdit(parent=self, label="Heading signal: ")
        self.propertiesGrid = TwoColumnsGrid(parent=parent, firstColumnName="Property name", secondColumnName="Property signal", buttonLabel="Add property")
        self.alertsGrid = TwoColumnsGrid(parent=parent, firstColumnName="Alerts name", secondColumnName="Alerts signals directory", buttonLabel="Add alerts directory")
        self.submitButton = QPushButton("Submit")
        layout = QVBoxLayout()
        layout.addWidget(self.name)
        layout.addWidget(self.connectionString)
        layout.addWidget(self.width)
        layout.addWidget(self.height)
        layout.addWidget(self.frontLidarRange)
        layout.addWidget(self.rearLidarRange)
        layout.addWidget(self.xSignal)
        layout.addWidget(self.ySignal)
        layout.addWidget(self.headingSignal)
        layout.addWidget(self.propertiesGrid)
        layout.addWidget(self.alertsGrid)
        layout.addWidget(self.submitButton)
        self.setLayout(layout)
        self.submitButton.clicked.connect(self.__onSubmit)
        self.__onSubmitCallback = onSubmit

    def assignObject(self, objectData):
        self.name.lineEdit.setText(objectData.name)
        self.connectionString.lineEdit.setText(objectData.connectionString)
        self.width.spinbox.setValue(float(objectData.width))
        self.height.spinbox.setValue(float(objectData.height))
        self.frontLidarRange.spinbox.setValue(float(objectData.frontLidarRange))
        self.rearLidarRange.spinbox.setValue(float(objectData.rearLidarRange))
        self.xSignal.lineEdit.setText(objectData.xSignal)
        self.ySignal.lineEdit.setText(objectData.ySignal)
        self.headingSignal.lineEdit.setText(objectData.headingSignal)
        self.propertiesGrid.fill(objectData.properties)
        self.alertsGrid.fill(objectData.alerts)

    def __onSubmit(self):
        objectData = ObjectData(name=self.name.lineEdit.text(),
                                sourceType=self.sourceType,
                                connectionString=self.connectionString.lineEdit.text(),
                                width=self.width.spinbox.value(),
                                height=self.height.spinbox.value(),
                                xSignal=self.xSignal.lineEdit.text(),
                                ySignal=self.ySignal.lineEdit.text(),
                                headingSignal=self.headingSignal.lineEdit.text(),
                                properties=self.propertiesGrid.dumpGrid(),
                                alerts=self.alertsGrid.dumpGrid(),
                                updateInterval=0.1,
                                frontLidarRange=self.frontLidarRange.spinbox.value(),
                                rearLidarRange=self.rearLidarRange.spinbox.value()
                                )
        self.__onSubmitCallback(objectData)
        self.close()


class ObjectsView(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.objectsTreeView = QTreeWidget(parent=self)
        self.objectsTreeView.setColumnCount(2)
        self.objectsTreeView.setHeaderLabels({"Option", "Value"})
        layout = QHBoxLayout()
        layout.addWidget(self.objectsTreeView)
        self.setLayout(layout)

    def addObject(self, objectData):
        objectRoot = self.__createObjectItem(objectData)
        self.objectsTreeView.insertTopLevelItem(self.objectsTreeView.topLevelItemCount(), objectRoot)

    def modifyObject(self, objectData, index):
        objectRoot = self.__createObjectItem(objectData)
        self.objectsTreeView.takeTopLevelItem(index)
        self.objectsTreeView.insertTopLevelItem(index, objectRoot)

    def objectDataFromItem(self, objectRootItem):
        objectData = ObjectData(name=objectRootItem.text(1),
                                sourceType=objectRootItem.child(0).text(1),
                                connectionString=objectRootItem.child(1).text(1),
                                width=objectRootItem.child(2).text(1),
                                height=objectRootItem.child(3).text(1),
                                xSignal=objectRootItem.child(6).text(1),
                                ySignal=objectRootItem.child(7).text(1),
                                headingSignal=objectRootItem.child(8).text(1),
                                properties=[],
                                alerts=[],
                                updateInterval=objectRootItem.child(11).text(1),
                                frontLidarRange=objectRootItem.child(4).text(1),
                                rearLidarRange=objectRootItem.child(5).text(1)
                                )
        propertiesRoot = objectRootItem.child(9)
        for i in range(0, propertiesRoot.childCount()):
            property = propertiesRoot.child(i).text(0), propertiesRoot.child(i).text(1)
            objectData.properties.append(property)

        alertsRoot = objectRootItem.child(10)
        for i in range(0, alertsRoot.childCount()):
            alert = alertsRoot.child(i).text(0), alertsRoot.child(i).text(1)
            objectData.alerts.append(alert)

        return objectData

    def __createObjectItem(self, objectData):
        objectRoot = self.__createItem(self.objectsTreeView, "Name", objectData.name)
        self.__createItem(objectRoot, "Source type", objectData.sourceType)
        self.__createItem(objectRoot, "Connection string", objectData.connectionString)
        self.__createItem(objectRoot, "Width", objectData.width)
        self.__createItem(objectRoot, "Height", objectData.height)
        self.__createItem(objectRoot, "Front lidar range", objectData.frontLidarRange)
        self.__createItem(objectRoot, "Rear lidar range", objectData.rearLidarRange)
        self.__createItem(objectRoot, "X coordinate signal", objectData.xSignal)
        self.__createItem(objectRoot, "Y coordinate signal", objectData.ySignal)
        self.__createItem(objectRoot, "Heading signal", objectData.headingSignal)
        propertiesRoot = self.__createItem(objectRoot, "Properties", "")
        for propertyItem in objectData.properties:
            self.__createItem(propertiesRoot, propertyItem[0], propertyItem[1])
        alertsRoot = self.__createItem(objectRoot, "Alerts", "")
        for alertItem in objectData.alerts:
            self.__createItem(alertsRoot, alertItem[0], alertItem[1])
        self.__createItem(objectRoot, "Update interval", objectData.updateInterval).setHidden(True)
        return objectRoot

    def __createItem(self, parent, firstColumn, secondColumn):
        item = QTreeWidgetItem(parent)
        item.setText(0, firstColumn)
        item.setText(1, str(secondColumn))
        return item


class AddObjectsWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.groupBox = QGroupBox(parent=self, title="Visualization objects")
        self.objectsView = ObjectsView(parent=self.groupBox)
        self.manageObjectWidget = ManageObjectWidget(parent=self.groupBox)
        groupboxLayout = QVBoxLayout()
        groupboxLayout.addWidget(self.objectsView)
        groupboxLayout.addWidget(self.manageObjectWidget)
        self.groupBox.setLayout(groupboxLayout)
        layout = QVBoxLayout()
        layout.addWidget(self.groupBox)
        self.setLayout(layout)
        self.manageObjectWidget.addObjectButton.clicked.connect(self.__addObject)
        self.manageObjectWidget.editObjectButton.clicked.connect(self.__editObject)
        self.manageObjectWidget.copyObjectButton.clicked.connect(self.__copyObject)
        self.manageObjectWidget.removeObjectButton.clicked.connect(self.__removeObject)
        self.objectsView.objectsTreeView.itemSelectionChanged.connect(self.__onObjectSelectionChanged)

    def onObjectAdded(self, objectData):
        self.objectsView.addObject(objectData)

    def onObjectEdited(self, objectData):
        selectedItem = self.objectsView.objectsTreeView.selectedItems()[0]
        self.objectsView.modifyObject(objectData, self.objectsView.objectsTreeView.indexOfTopLevelItem(selectedItem))

    def __addObject(self):
        dialog = AddObjectDialog(parent=self, sourceType=self.manageObjectWidget.pickObjectTypeCombo.currentText(), onSubmit=self.onObjectAdded)
        dialog.exec()

    def __copyObject(self):
        self.onObjectAdded(self.objectsView.objectDataFromItem(self.objectsView.objectsTreeView.selectedItems()[0]))

    def __editObject(self):
        selectedItem = self.objectsView.objectsTreeView.selectedItems()[0]
        dialog = AddObjectDialog(parent=self, sourceType=self.manageObjectWidget.pickObjectTypeCombo.currentText(), onSubmit=self.onObjectEdited)
        dialog.assignObject(self.objectsView.objectDataFromItem(selectedItem))
        dialog.exec()

    def __removeObject(self):
        self.objectsView.objectsTreeView.takeTopLevelItem(self.objectsView.objectsTreeView.indexOfTopLevelItem(self.objectsView.objectsTreeView.selectedItems()[0]))

    def __onObjectSelectionChanged(self):
        items = self.objectsView.objectsTreeView.selectedItems()
        if len(items) > 0:
            selectedItem = items[0]
            parent = selectedItem.parent()
            self.manageObjectWidget.editObjectButton.setEnabled(parent is None)
            self.manageObjectWidget.copyObjectButton.setEnabled(parent is None)
            self.manageObjectWidget.removeObjectButton.setEnabled(parent is None)

    def objects(self):
        objects = list()
        for i in range(0, self.objectsView.objectsTreeView.topLevelItemCount()):
            objectRoot = self.objectsView.objectsTreeView.topLevelItem(i)
            objects.append(self.objectsView.objectDataFromItem(objectRoot))
        return objects


class ConfigurationWizard(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.__configurationPersistency = None
        self.mapDataWidget = MapDataWidget(self)
        self.addObjectsWidget = AddObjectsWidget(self)
        self.submitButton = QPushButton(parent=self, text="Submit")
        self.cancelButton = QPushButton(parent=self, text="Cancel")
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch()
        buttonsLayout.addWidget(self.submitButton)
        buttonsLayout.addWidget(self.cancelButton)
        layout = QVBoxLayout()
        layout.addWidget(self.mapDataWidget)
        layout.addWidget(self.addObjectsWidget)
        layout.addLayout(buttonsLayout)
        self.setLayout(layout)
        self.accepted.connect(self.__onDialogAccepted)
        self.submitButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

    def driveConfiguration(self, persistency):
        self.__configurationPersistency = persistency
        self.exec()

    def driveEditConfiguration(self, persistency):
        self.__configurationPersistency = persistency
        widgetsBuilder = ConfigurationWidgetsBuilder(persistency, self.mapDataWidget, self.addObjectsWidget)
        widgetsBuilder.build()
        self.exec()

    def __onDialogAccepted(self):
        self.__configurationPersistency.saveMapData(self.mapDataWidget.mapData())
        self.__configurationPersistency.saveObjects(self.addObjectsWidget.objects())
        self.__configurationPersistency.write()