

class AbstractUserView:
    def __init__(self):
        pass

    def driveConfigCreation(self, persistency):
        raise Exception("Not implemented!")

    def driveConfigEdit(self, persistency):
        raise Exception("Not implemented!")

    def askForConfigPath(self):
        raise Exception("Not implemented!")

    def onIncorrectConfig(self, configFile):
        raise Exception("Not implemented!")
