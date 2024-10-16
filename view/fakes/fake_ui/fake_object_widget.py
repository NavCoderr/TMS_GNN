class FakeObjectWidget:
    def __init__(self, widgetLogic):
        self.__widgetLogic = widgetLogic

    def x(self):
        return self.__widgetLogic.x()

    def y(self):
        return self.__widgetLogic.y()

