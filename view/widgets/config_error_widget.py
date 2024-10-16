from PyQt6.QtWidgets import *


class IncorrectConfigWrapper:
    def __init__(self):
        pass

    def onIncorrectConfig(self, configFile):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("Incorrect configuration file")
        msg.setInformativeText("Configuration file {} is not a valid configuration file".format(configFile))
        msg.setWindowTitle("Configuration error")
        msg.exec()

        