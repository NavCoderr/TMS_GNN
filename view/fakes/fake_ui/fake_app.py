from view.fakes.fake_ui.fake_mainframe import Mainframe
from view.fakes.fake_events_hub import FakeBusinessRules


class FakeApp:
    def __init__(self, mapSize, updatesGenerator):
        testObjectsNum = 1
        self.__businessRules = FakeBusinessRules(range(0, testObjectsNum), mapSize, updatesGenerator)
        self.mainframe = Mainframe(self.__businessRules)

