import threading
import time

from model.abstract_event_source import AbstractEventSource
from model.events import *

CHILD_SIGNALS_REFRESH_COUNT=1000


class PollingThread:
    def __init__(self, client, signalsList, callback, pollingInterval, errorSink, connectionString, alertsSignalsRoots, eventsSource):
        self.__thread = None
        self.__client = client
        self.__signalsList = signalsList
        self.__callback = callback
        self.__stopped = False
        self.__pollingInterval = pollingInterval
        self.__errorSink = errorSink
        self.__connectionString = connectionString
        self.__connected = False
        self.__alertsSignalsRoots = alertsSignalsRoots
        self.__eventsSource = eventsSource
        self.__childSignalsRefreshCountdown = 0

    def addSignal(self, signal):
        self.__signalsList.append(signal)

    def start(self):
        self.__thread = threading.Thread(target=self.__pollingMethod)
        self.__stopped = False
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        self.__client.disconnect()
        self.__connected = False
        self.__stopped = True
        self.__thread.join()
        self.__thread = None

    def __pollingMethod(self):
        while not self.__stopped:
            try:
                self.__childSignalsRefreshCountdown -= 1
                self.__ensureConnection()
                signalsDict = {}
                for signal in self.__signalsList:
                    if len(signal) > 0:
                        signalsDict[str(signal)] = self.__client.getSignalValue(signal)
                self.__callback(signalsDict)
                if self.__childSignalsRefreshCountdown <= 0:
                    self.__childSignalsRefreshCountdown = CHILD_SIGNALS_REFRESH_COUNT
                    self.__refreshAlerts()
                else:
                    time.sleep(self.__pollingInterval)
            except Exception as e:
                self.__errorSink.logDebug("Exception during polling: " + str(e))
            except:
                self.__errorSink.logDebug("OpcEventsSource PollingThread: ignored unhandled exception")
                pass

    def __ensureConnection(self):
        if not self.__connected:
            timeoutPeriod = 5 * 60
            try:
                self.__client.connect(self.__connectionString)
                self.__connected = True
            except Exception as e:
                self.__errorSink.logInformation("Cannot connect to: {}, entering wait for: {} s".format(self.__connectionString, timeoutPeriod))
                self.__enterTimoutPeriod(timeoutPeriod)
            except:
                self.__errorSink.logDebug("Unhandled exception while connection, killing opc object")
                self.stop()

    def __enterTimoutPeriod(self, timeoutPeriod):
        while not self.__connected and not self.__stopped and timeoutPeriod > 0:
            __sleepTime = 1
            time.sleep(__sleepTime)
            timeoutPeriod -= __sleepTime

    def __refreshAlerts(self):
        for alertRoot in self.__alertsSignalsRoots:
            signals = self.__getAlertsSignalsForRoot(alertRoot)
            for signalName in signals:
                self.__eventsSource.addAlertSignal(signalName, signals[signalName], alertRoot)

    def __getAlertsSignalsForRoot(self, alertRoot):
        return self.__client.getChildSignals(self.__alertsSignalsRoots[alertRoot])


class OpcEventSource(AbstractEventSource):
    def __init__(self, opcClient, objectId, xSignal, ySignal, rotationSignal, propertiesSignalsDict, updateInterval, errorSink, connectionString, alertsSignals):
        super().__init__()
        signalsList = [xSignal, ySignal, rotationSignal]
        self.__handlers = dict()
        self.__opcClient = opcClient
        self.__id = objectId
        self.__xSignalStr = str(xSignal)
        self.__ySignalStr = str(ySignal)
        self.__rotationSignalStr = str(rotationSignal)
        self.__currentSignalsState = {self.__xSignalStr: 0, self.__ySignalStr: 0, self.__rotationSignalStr: 0}
        self.__propertiesSignalsStrings = dict()
        self.__alertsSignalsStrings = dict()
        self.__alertsParents = dict()
        for propertyName in propertiesSignalsDict:
            signal = propertiesSignalsDict[propertyName]
            self.__propertiesSignalsStrings[propertyName] = str(signal)
            self.__currentSignalsState[str(signal)] = 0
            signalsList.append(signal)
        self.__pollingThread = PollingThread(opcClient, signalsList, self.__dataPolledCallback, updateInterval, errorSink, connectionString, alertsSignals, self)

    def addHandler(self, handler):
        self.__handlers[id(handler)] = handler

    def removeHandler(self, handler):
        del self.__handlers[id(handler)]

    def addAlertSignal(self, alertSignalName, alertSignal, alertParent):
        if alertSignalName not in self.__alertsSignalsStrings:
            self.__pollingThread.addSignal(alertSignal)
            self.__alertsSignalsStrings[alertSignalName] = str(alertSignal)
            self.__alertsParents[alertSignalName] = alertParent
        self.__currentSignalsState[str(alertSignal)] = None

    def start(self):
        self.__pollingThread.start()

    def stop(self):
        self.__pollingThread.stop()

    def sendRegisterObjectEvent(self, type, properties, width, height, name, frontLidarRange, rearLidarRange):
        registerObjectEvent = RegisterObjectEvent(objectId=self.__id,
                                                  type=type,
                                                  properties=properties,
                                                  width=width,
                                                  height=height,
                                                  name=name,
                                                  frontLidarRange=frontLidarRange,
                                                  rearLidarRange=rearLidarRange
                                                  )
        self.__broadcastEvent(registerObjectEvent)

    def __broadcastEvent(self, event):
        for handlerId in self.__handlers:
            self.__handlers[handlerId].onEvent(event)

    def __dataPolledCallback(self, signalsDict):
        self.__signalsDictToEvent(signalsDict)

    def __signalsDictToEvent(self, signalsDict):
        self.__processPositionSignals(signalsDict)
        self.__processRotationSignal(signalsDict)
        self.__processPropertiesSignals(signalsDict)
        self.__processAlertsSignals(signalsDict)

    def __processPositionSignals(self, signalsDict):
        newX = None
        newY = None
        if signalsDict[self.__xSignalStr] != self.__currentSignalsState[self.__xSignalStr]:
            newX = signalsDict[self.__xSignalStr]
        if signalsDict[self.__ySignalStr] != self.__currentSignalsState[self.__ySignalStr]:
            newY = signalsDict[self.__ySignalStr]
        if newX or newY:
            self.__onPositionChanged(newX, newY)

    def __onPositionChanged(self, newX, newY):
        if newX:
            self.__currentSignalsState[self.__xSignalStr] = newX
        if newY:
            self.__currentSignalsState[self.__ySignalStr] = newY
        self.__broadcastEvent(UpdateObjectPositionEvent(objectId=self.__id,
                                                        x=self.__currentSignalsState[self.__xSignalStr],
                                                        y=self.__currentSignalsState[self.__ySignalStr]))

    def __processPropertiesSignals(self, signalsDict):
        changed = False
        for propertyName in self.__propertiesSignalsStrings:
            signalStr = self.__propertiesSignalsStrings[propertyName]
            if signalsDict[signalStr] != self.__currentSignalsState[signalStr]:
                self.__currentSignalsState[signalStr] = signalsDict[signalStr]
                changed = True
        if changed:
            self.__onPropertiesChanged()

    def __onPropertiesChanged(self):
        self.__broadcastEvent(UpdateObjectPropertiesEvent(objectId=self.__id, properties=self.__getPropertiesDict()))

    def __getPropertiesDict(self):
        res = dict()
        for propertyName in self.__propertiesSignalsStrings:
            signalStr = self.__propertiesSignalsStrings[propertyName]
            signalValue = self.__currentSignalsState[signalStr]
            res[propertyName] = signalValue
        return res

    def __processAlertsSignals(self, signalsDict):
        changedAlerts = list()
        for alertName in self.__alertsSignalsStrings:
            signalStr = self.__alertsSignalsStrings[alertName]
            if signalsDict[signalStr] != self.__currentSignalsState[signalStr]:
                self.__currentSignalsState[signalStr] = signalsDict[signalStr]
                changedAlerts.append(alertName)
        if len(changedAlerts) > 0:
            self.__onAlertsChanged(changedAlerts)

    def __onAlertsChanged(self, changedAlerts):
        self.__broadcastEvent(UpdateObjectAlertsEvent(objectId=self.__id, alerts=self.__getChangedAlertsDict(changedAlerts)))

    def __getChangedAlertsDict(self, changedAlerts):
        res = dict()
        for alertName in changedAlerts:
            signalStr = self.__alertsSignalsStrings[alertName]
            signalValue = self.__currentSignalsState[signalStr]
            res[self.__alertsParents[alertName]] = {alertName: signalValue}
        return res

    def __processRotationSignal(self, signalsDict):
        newRotation = signalsDict[self.__rotationSignalStr]
        if newRotation != self.__currentSignalsState[self.__rotationSignalStr]:
            self.__currentSignalsState[self.__rotationSignalStr] = newRotation
            self.__onRotationChanged(newRotation)

    def __onRotationChanged(self, newRotation):
        self.__broadcastEvent(UpdateObjectRotationEvent(objectId=self.__id, rotation=newRotation))