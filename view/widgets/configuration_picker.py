from PyQt6.QtWidgets import *
from PyQt6.QtCore import *


CONFIG_EXTENSION = "json"


class ConfigurationPickerDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.__editRequested = False
        self.setWindowTitle("Select configuration file")
        self.lineEdit = QLineEdit()
        self.browseButton = QPushButton("Browse")
        self.newButton = QPushButton("New")
        self.startButton = QPushButton("Start")
        self.editButton = QPushButton("Edit")
        layout = QHBoxLayout()
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.browseButton)
        layout.addWidget(self.newButton)
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addStretch()
        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(self.editButton)
        buttonLayout.addWidget(self.startButton)
        mainLayout.addLayout(buttonLayout)
        self.setLayout(mainLayout)
        self.browseButton.clicked.connect(self.__showBrowseDialog)
        self.newButton.clicked.connect(self.__showNewConfigDialog)
        self.startButton.clicked.connect(self.__onApply)
        self.editButton.clicked.connect(self.__onEditConfiguration)

    def getConfigurationPath(self):
        screenGeometry = QApplication.primaryScreen().geometry()
        windowGeometry = QRect()
        width = 600
        height = 400
        windowGeometry.setLeft(int(screenGeometry.width() / 2) - width)
        windowGeometry.setTop(int(screenGeometry.height() / 2) - height)
        windowGeometry.setWidth(width)
        windowGeometry.setHeight(height)
        self.setGeometry(windowGeometry)
        self.exec()
        return self.lineEdit.text(), self.__editRequested

    def __showBrowseDialog(self):
        __file = QFileDialog.getOpenFileName(parent=self,
                                                  caption="Open file",
                                                  directory="",
                                                  filter="Configuration files (*.{})".format(CONFIG_EXTENSION))[0]
        self.lineEdit.setText(__file)

    def __showNewConfigDialog(self):
        __file = QFileDialog.getSaveFileName(parent=self,
                                                  caption="Create file",
                                                  directory="",
                                                  filter="Configuration files (*.{})".format(CONFIG_EXTENSION))[0]
        self.lineEdit.setText(__file)

    def __onApply(self):
        self.close()

    def __onEditConfiguration(self):
        self.__editRequested = True
        self.__onApply()
