from view.fakes.fake_ui.fake_object_widget import FakeObjectWidget


class ViewSize:
    def __init__(self, size):
        self.__size = size

    def width(self):
        return self.__size[0]

    def height(self):
        return self.__size[1]


class FakeMapWidget:
    def __init__(self, widgetLogic):
        widgetLogic.setViewAccess(self)
        self.__size = None
        self.__objectsWidgets = list()

    def setSize(self, size):
        self.__size = ViewSize(size)

    def size(self):
        return self.__size

    def updateGrid(self, width, height):
        pass

    def showMap(self):
        pass

    def addObject(self, viewObject):
        self.__objectsWidgets.append(FakeObjectWidget(viewObject))

    def objectWidget(self, index):
        return self.__objectsWidgets[index]

    def updateView(self):
        pass
