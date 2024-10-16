import random
import threading
import time
from opc_adapter.opc_client_factory import AbstractOpcClientFactory

xSignal = ['x']
ySignal = ['y']
headingSignal = ['heading']
batterySignal = ['battery']
warningSignal1 = ['warning1']
warningSignal2 = ['warning2']
informationSignal1 = ['information1']
informationSignal2 = ['information2']
childSignals = {
    "['warnings_root']" : { "/".join(warningSignal1): warningSignal1, "/".join(warningSignal2): warningSignal2 },
    "['information_root']" : { "/".join(informationSignal1): informationSignal1, "/".join(informationSignal2): informationSignal2 },
}
KnownSignals = [
    str(xSignal),
    str(ySignal),
    str(headingSignal),
    str(batterySignal),
    str(warningSignal1),
    str(warningSignal2),
    str(informationSignal1),
    str(informationSignal2)]
FailureProbability = 0.1  # how often communication with serve will fail
MinX = -5  # outside the map origin
MinY = -5  # outside the map origin
MaxX = 30  # a bit larger than map width
MaxY = 55  # a bit larger than map height
UpdateInterval = 0.1  # s
ConnectionTime = 1  # s
Steps = 25
RotationSteps = 7


class FakeOpcClientFactory(AbstractOpcClientFactory):
    def __init__(self):
        super().__init__()
        pass

    def createOpcClient(self, errorSink):
        return FakeOpcClient(errorSink)


class ObjectState:
    def __init__(self):
        self.__signals = dict()
        self.__signals[str(xSignal)] = MinX
        self.__signals[str(ySignal)] = MinY
        self.__signals[str(headingSignal)] = 0
        self.__signals[str(batterySignal)] = 0
        self.__signals[str(warningSignal1)] = False
        self.__signals[str(warningSignal2)] = True
        self.__signals[str(informationSignal1)] = False
        self.__signals[str(informationSignal2)] = True

    def getSignalValue(self, signal):
        if str(signal) in KnownSignals:
            return self.__signals[str(signal)]
        raise Exception("Unknown signal!")

    def setSignalValue(self, signal, value):
        if str(signal) in KnownSignals:
            self.__signals[str(signal)] = value
        else:
            raise Exception("Unknown signal!")


class StrategyBase:
    def __init__(self, objectState):
        self.__thread = threading.Thread(target=self.__updateMethod)
        self.__thread.daemon = True
        self._objectState = objectState
        self.__updateFailureProbability = FailureProbability * (0.5 + random.random())

    def start(self):
        self.__thread.start()

    def __updateMethod(self):
        while True:
            self._onBeforeUpdate()
            self._objectState.setSignalValue(xSignal, self._updateX())
            self._objectState.setSignalValue(ySignal, self._updateY())
            self._objectState.setSignalValue(headingSignal, self._updateHeading())
            self._objectState.setSignalValue(batterySignal, random.randint(0, 100))
            self._objectState.setSignalValue(warningSignal1, random.randint(0, 1) == 1)
            self._objectState.setSignalValue(informationSignal1, random.randint(0, 1) == 1)
            time.sleep((1+random.random()) * UpdateInterval)
            if random.random() <= self.__updateFailureProbability:
                time.sleep((1+random.random()) * UpdateInterval * 10)

    def _updateX(self):
        raise Exception("Not Implemented")

    def _updateY(self):
        raise Exception("Not Implemented")

    def _onBeforeUpdate(self):
        pass

    def _updateHeading(self):
        return random.random()


class HorizontalMovingStrategy(StrategyBase):

    def __init__(self, objectState):
        super().__init__(objectState)
        self.__nextUpdateY = False

    def _updateX(self):
        step = (MaxX - MinX) / Steps
        res = self._objectState.getSignalValue(xSignal) + step
        if res > MaxX:
            self.__nextUpdateY = True
            return MinX
        return res

    def _updateY(self):
        y = self._objectState.getSignalValue(ySignal)
        if self.__nextUpdateY:
            self.__nextUpdateY = False
            step = (MaxY - MinY) / Steps
            res = y + step
            if res > MaxY:
                return MinY
            return res
        return y


class VerticalMovingStrategy(StrategyBase):
    def __init__(self, objectState):
        super().__init__(objectState)
        self.__nextUpdateX = False

    def _updateX(self):
        x = self._objectState.getSignalValue(xSignal)
        if self.__nextUpdateX:
            self.__nextUpdateX = False
            step = (MaxX - MinX) / Steps
            res = x + step
            if res > MaxX:
                return MinX
            return res
        return x

    def _updateY(self):
        step = (MaxY - MinY) / Steps
        res = self._objectState.getSignalValue(ySignal) + step
        if res > MaxY:
            self.__nextUpdateX = True
            return MinY
        return res


class PointsMovingStrategy(StrategyBase):
    Points = [
        (MinX + ((MaxX - MinX) * 0.25), MinY + ((MaxY - MinY) * 0.25)),
        (MinX + ((MaxX - MinX) * 0.75), MinY + ((MaxY - MinY) * 0.25)),
        (MinX + ((MaxX - MinX) * 0.75), MinY + ((MaxY - MinY) * 0.75)),
        (MinX + ((MaxX - MinX) * 0.25), MinY + ((MaxY - MinY) * 0.75))
    ]

    Headings = [ 180, 90, 0, 270 ]

    def __init__(self, objectState):
        super().__init__(objectState)
        self.__x = self.Points[0][0]
        self.__y = self.Points[0][1]
        self.__assignPoint(1)

    def _onBeforeUpdate(self):
        if self.__rotationSteps > 0:
            self.__updateRotationBeforeUpdate()
        else:
            self.__updatePositionBeforeUpdate()

    def _updateX(self):
        return self.__x

    def _updateY(self):
        return self.__y

    def _updateHeading(self):
        return self.__rotation

    def __assignPoint(self, index):
        self.__targetIndex = index
        self.__targetPoint = self.Points[self.__targetIndex]
        self.__steps = Steps
        self.__rotation = self.Headings[self.__targetIndex - 1]
        self.__targetRotation = self.Headings[self.__targetIndex]
        self.__rotationSteps = RotationSteps
        d1 = (self.__targetRotation - self.__rotation)
        d2 = (360 - d1)
        self.__rotationIncrement = min(d1, d2) / self.__rotationSteps
        if d1 > d2:
            self.__rotationIncrement *= -1

    def __updatePositionBeforeUpdate(self):
        xDistance = self.__targetPoint[0] - self.__x
        yDistance = self.__targetPoint[1] - self.__y
        stepExecuted = False
        if abs(xDistance) > 0:
            self.__x += (xDistance / self.__steps)
            stepExecuted = True
        if abs(yDistance) > 0:
            self.__y += (yDistance / self.__steps)
            stepExecuted = True
        if stepExecuted:
            self.__steps -= 1
        if abs(xDistance) == 0 and abs(yDistance) == 0:
            self.__targetIndex += 1
            if self.__targetIndex >= len(self.Points):
                self.__targetIndex = 0
            self.__assignPoint(self.__targetIndex)

    def __updateRotationBeforeUpdate(self):
        self.__rotation += self.__rotationIncrement
        self.__rotationSteps -= 1


class FakeOpcClient:
    MovingStrategies = [
#        HorizontalMovingStrategy,
#        VerticalMovingStrategy,
        PointsMovingStrategy,
    ]  # possible map traversing algorithms

    def __init__(self, errorSink):
        self.__connected = False
        self.__errorSink = errorSink
        self.__objectState = ObjectState()
        self.__strategy = random.choice(self.MovingStrategies)(self.__objectState)
        self.__strategy.start()

    def connect(self, connectionString):
        time.sleep(ConnectionTime)
        if random.random() <= (FailureProbability * 5):
            self.__throw()
        self.__connected = True

    def disconnect(self):
        self.__connected = False

    def getSignalValue(self, signal):
        if random.random() <= FailureProbability:
            self.__throw()
        return self.__objectState.getSignalValue(signal)

    def getChildSignals(self, root):
        return childSignals[str(root)]

    def __throw(self):
        raise Exception("Server failure")

