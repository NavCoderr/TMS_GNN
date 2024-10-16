class ErrorSink:
    def __init__(self):
        self.listeners = dict()

    def addListener(self, listener):
        self.listeners[id(listener)] = listener

    def removeListener(self, listener):
        self.listeners.pop(id(listener))

    def logError(self, errorMessage):
        for listenerId in self.listeners:
            self.listeners[listenerId].logError(errorMessage)

    def logWarning(self, errorMessage):
        for listenerId in self.listeners:
            self.listeners[listenerId].logWarning(errorMessage)

    def logInformation(self, errorMessage):
        for listenerId in self.listeners:
            self.listeners[listenerId].logInformation(errorMessage)

    def logDebug(self, errorMessage):
        for listenerId in self.listeners:
            self.listeners[listenerId].logDebug(errorMessage)

