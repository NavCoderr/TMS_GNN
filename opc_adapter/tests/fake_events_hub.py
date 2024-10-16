from model.abstract_event_handler import AbstractEventHandler


class FakeEventsHub(AbstractEventHandler):
    def __init__(self):
        super().__init__()
        self.events = []

    def onEvent(self, event):
        self.events.append(event)

    def lastEvent(self):
        return self.events[len(self.events) - 1]
