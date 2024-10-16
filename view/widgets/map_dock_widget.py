from PyQt6.QtWidgets import *
from view.widgets.map_pane import MapPane


class MapDockWidget(QDockWidget):
    def __init__(self, parent, startAppCallback, mapWidgetLogic):
        super().__init__(parent=parent)
        self.mapPane = MapPane(parent=self, startAppCallback=startAppCallback, mapWidgetLogic=mapWidgetLogic)
        self.setWidget(self.mapPane)
        self.setWindowTitle("Visualization")
