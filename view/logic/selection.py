class Selection:

    def __init__(self):
        self.__selectedObject = None
        self.__selectionListeners = []

    def updateSelection(self, newSelection):
        if self.__selectedObject:
            self.__selectedObject.removeObserver(self)
        self.__selectedObject = newSelection
        if self.__selectedObject:
            self.__selectedObject.addObserver(self)
        self.__broadcastSelectedObjectChanged()

    def selectedObject(self):
        return self.__selectedObject

    def addListener(self, listener):
        self.__selectionListeners.append(listener)

    def objectChanged(self, object):
        self.__broadcastSelectedObjectChanged()

    def __broadcastSelectedObjectChanged(self):
        for listener in self.__selectionListeners:
            listener.onSelectionChanged()

