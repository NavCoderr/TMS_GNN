from PyQt6.QtWidgets import *
from PyQt6 import QtCore
from view.widgets.map_widget import MapWidget


class MapPane(QWidget):
    def __init__(self, parent, startAppCallback, mapWidgetLogic):
        super().__init__(parent=parent)
        self.scrollArea = QScrollArea(parent=self)
        self.scrollAreaWidget = QWidget(parent=self.scrollArea)

        self.startVisualizationButton = QPushButton(parent=self)
        self.startVisualizationButton.setText("Start")

        self.mapWidget = MapWidget(parent=self, widgetLogic=mapWidgetLogic)
        upperLayout = QHBoxLayout()
        upperLayout.addWidget(self.startVisualizationButton)
        upperLayout.addStretch()
        layout = QVBoxLayout()
        layout.addLayout(upperLayout)
        layout.addWidget(self.mapWidget)
        layout.addStretch()

        self.scrollAreaWidget.setLayout(layout)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.scrollAreaWidget)

        layout = QVBoxLayout()
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

        self.startVisualizationButton.clicked.connect(startAppCallback)

    def sizeHint(self) -> QtCore.QSize:
        return self.mapWidget.sizeHint() + super().sizeHint()