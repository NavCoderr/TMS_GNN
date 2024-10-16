from business_rules.abstract_user_view import AbstractUserView


class QtViewToAbstractUserView(AbstractUserView):
    def __init__(self):
        super().__init__()
        self.__configurationWizard = None
        self.__configurationPicker = None
        self.__incorrectConfigDialog = None

    def driveConfigCreation(self, persistency):
        self.__configurationWizard.driveConfiguration(persistency)

    def driveConfigEdit(self, persistency):
        self.__configurationWizard.driveEditConfiguration(persistency)

    def askForConfigPath(self):
        return self.__configurationPicker.getConfigurationPath()

    def onIncorrectConfig(self, configFile):
        self.__incorrectConfigDialog.onIncorrectConfig(configFile)

    def setConfigurationWizard(self, configurationWizard):
        self.__configurationWizard = configurationWizard

    def setConfigurationPicker(self, configurationPicker):
        self.__configurationPicker = configurationPicker

    def setIncorrectConfigDialog(self, incorrectConfigDialog):
        self.__incorrectConfigDialog = incorrectConfigDialog