from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *


class GridWidget(QWidget):

    def __init__(self, parent, columnWidth, rowHeight):
        super().__init__(parent=parent)
        self.columnWidth = columnWidth
        self.rowHeight = rowHeight

    def setColumnWidth(self, columnWidth):
        self.columnWidth = columnWidth

    def setRowHeight(self, rowHeight):
        self.rowHeight = rowHeight

    def paintGrid(self, painter):
        self.__paintRows(painter)
        self.__paintColumns(painter)

    def __paintRows(self, painter):
        count = self.__calculateRowsCount()
        pointBegin = QPoint(0, 0)
        pointEnd = QPoint(self.parent().width(), 0)
        for i in range(0, count+1):
            self.__drawLine(pointBegin, pointEnd, painter)
            pointBegin.setY(pointBegin.y() + self.rowHeight)
            pointEnd.setY(pointEnd.y() + self.rowHeight)

    def __paintColumns(self, painter):
        count = self.__calculateColumnsCounts()
        pointBegin = QPoint(0, 0)
        pointEnd = QPoint(0, self.parent().height())
        for i in range(0, count+1):
            self.__drawLine(pointBegin, pointEnd, painter)
            pointBegin.setX(pointBegin.x() + self.columnWidth)
            pointEnd.setX(pointEnd.x() + self.columnWidth)

    def __drawLine(self, pointBegin, pointEnd, painter):
        brush = QBrush()
        brush.setColor(QColor(0, 0, 0))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        pen = QPen()
        pen.setBrush(brush)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(QLine(pointBegin, pointEnd))


    def __calculateRowsCount(self):
        return int(self.parent().height() / self.rowHeight)

    def __calculateColumnsCounts(self):
        return int(self.parent().width() / self.columnWidth)


