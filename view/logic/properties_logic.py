class PropertiesLogic:
    def __init__(self, selection):
        self.selection = selection
        self.viewAccess = None
        self.selection.addListener(self)

    def setViewAccess(self, viewAccess):
        self.viewAccess = viewAccess

    def onSelectionChanged(self):
        if self.selection.selectedObject() is not None:
            self.viewAccess.setProperties(self.selection.selectedObject().properties())

