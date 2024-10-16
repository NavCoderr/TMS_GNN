from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6 import QtCore
from view.logic.map_widget_logic import VisualizationWidgetLogic


class VisualizationObjectWidget(QWidget):
    def __init__(self, parent, widgetLogic : VisualizationWidgetLogic):
        super().__init__(parent=parent)
        self.__widgetLogic = widgetLogic
        self.__hovered = False

    def __hoverLeaveEvent(self, e):
        self.__hovered = False

    def __hoverEnterEvent(self, e):
        self.__hovered = True

    def __onLeftMouseButtonReleased(self, e):
        self.__widgetLogic.updateSelection()

    def __onRightMouseButtonReleased(self, e):
        qmenu = QMenu(self.__widgetLogic.name(), self)
        disconnectAction = qmenu.addAction("Disconnect")
        disconnectAction.triggered.connect(self.__widgetLogic.disconnect)
        reconnectAction = qmenu.addAction("Reconnect")
        reconnectAction.triggered.connect(self.__widgetLogic.reconnect)
        point = e.globalPosition().toPoint()
        qmenu.popup(point)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.__onLeftMouseButtonReleased(e)
        elif e.button() == Qt.MouseButton.RightButton:
            self.__onRightMouseButtonReleased(e)
        return super().mousePressEvent(e)

    def paintObject(self, painter):
        self.__drawObjectBorder(painter)
        self.__drawObjectShape(painter)

    def event(self, e: QtCore.QEvent) -> bool:
        res = super().event(e)
        if e.type() == QEvent.Type.HoverEnter:
            self.__hoverEnterEvent(e)
        elif e.type() == QEvent.Type.HoverLeave:
            self.__hoverLeaveEvent(e)
        return res

    def rect(self) -> QtCore.QRect:
        return QtCore.QRect(*self.__widgetLogic.getBoundingRect())

    def __drawObjectBorder(self, painter):
        frontEllipse = QtCore.QRect(*self.__widgetLogic.getFrontLidarEllipseRect())
        rearEllipse = QtCore.QRect(*self.__widgetLogic.getRearLidarEllipseRect())
        color = QColor(0, 0, 0, 50)

        backgroundBrush = QBrush()
        backgroundBrush.setColor(color)
        backgroundBrush.setStyle(Qt.BrushStyle.SolidPattern)
        borderBrush = QBrush()
        borderBrush.setColor(QColor(0, 0, 0))
        borderBrush.setStyle(Qt.BrushStyle.SolidPattern)
        pen = QPen()
        pen.setBrush(borderBrush)
        pen.setWidth(5)
        painter.setBrush(backgroundBrush)
        painter.setPen(pen)
        painter.drawEllipse(frontEllipse)
        painter.drawEllipse(rearEllipse)

    def __drawObjectShape(self, painter):
        color = QColor(20, 20, 120)
        if self.__widgetLogic.isSelected():
            color = QColor(20, 120, 0)
        borderBrush = QBrush()
        borderBrush.setColor(QColor(0, 0, 0))
        borderBrush.setStyle(Qt.BrushStyle.SolidPattern)
        backgroundBrush = QBrush()
        backgroundBrush.setColor(color)
        backgroundBrush.setStyle(Qt.BrushStyle.SolidPattern)
        polygon = QPolygon(self.__pointsToQPoints(self.__widgetLogic.getShapePoints()))
        pen = QPen()
        pen.setBrush(borderBrush)
        pen.setWidth(5)
        painter.setPen(pen)
        painter.drawPolygon(polygon)
        painterPath = QPainterPath()
        painterPath.addPolygon(polygon.toPolygonF())
        painter.fillPath(painterPath, backgroundBrush)

    def __pointsToQPoints(self, points):
        qpoints = []
        for point in points:
            qpoints.append(QPoint(point.x, point.y))
        return qpoints


