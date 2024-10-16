class FakeOutputWidget:
    def __init__(self, widgetLogic):
        widgetLogic.setViewAccess(self)

