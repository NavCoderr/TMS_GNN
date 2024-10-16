class FakeErrorSink:
    def __init__(self):
        self.errors = list()

    def logError(self, msg):
        self.errors.append(msg)

