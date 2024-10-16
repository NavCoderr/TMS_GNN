import copy
from dataclasses import dataclass


@dataclass
class FrameField:
    startingByte: int
    length: int
    type: type
    endianStyle: str
    name: str


dataLengthByFrameId = {
    5000: 48,
    6000: 188,
    6100: 134
}


VARIABLE_LENGTH_MARK = 100000


class DtlDateTime:
    def __init__(self, year, month, day, dayOfWeek, hour, minute, second, millisecond):
        self.year = year
        self.month = month
        self.day = day
        self.dayOfWeek = dayOfWeek
        self.hour = hour
        self.minute = minute
        self.second = second
        self.millisecond = millisecond

    @staticmethod
    def from_bytes(_bytes, byteorder):
        year = int.from_bytes(_bytes[0: 2], byteorder)
        month = int.from_bytes(_bytes[2: 3], byteorder)
        day = int.from_bytes(_bytes[3: 4], byteorder)
        dayOfWeek = int.from_bytes(_bytes[4: 5], byteorder)
        hour = int.from_bytes(_bytes[5: 6], byteorder)
        minute = int.from_bytes(_bytes[6:7], byteorder)
        second = int.from_bytes(_bytes[7:8], byteorder)
        millisecond = int.from_bytes(_bytes[8: 12], byteorder)
        return DtlDateTime(year, month, day, dayOfWeek, hour, minute, second, millisecond)

    def to_bytes(self, unusedLength, byteorder):
        res = bytearray()
        res.extend(self.year.to_bytes(2, byteorder))
        res.extend(self.month.to_bytes(1, byteorder))
        res.extend(self.day.to_bytes(1, byteorder))
        res.extend(self.dayOfWeek.to_bytes(1, byteorder))
        res.extend(self.hour.to_bytes(1, byteorder))
        res.extend(self.minute.to_bytes(1, byteorder))
        res.extend(self.second.to_bytes(1, byteorder))
        res.extend(self.millisecond.to_bytes(4, byteorder))
        return bytes(res)


class ProductionOrderId:
    def __init__(self, orderType,
                 transportedGood,
                 customerId,
                 sourcePoint,
                 destinationPoint,
                 dayOfGeneration,
                 monthOfGeneration,
                 yearOfGeneration
                 ):
        self.orderType = orderType
        self.transportedGood = transportedGood
        self.customerId = customerId
        self.sourcePoint = sourcePoint
        self.destinationPoint = destinationPoint
        self.dayOfGeneration = dayOfGeneration
        self.monthOfGeneration = monthOfGeneration
        self.yearOfGeneration = yearOfGeneration

    @staticmethod
    def from_bytes(_bytes, byteorder):
        orderType = int.from_bytes(_bytes[0: 2], byteorder)
        transportedGood = int.from_bytes(_bytes[2: 4], byteorder)
        customerId = int.from_bytes(_bytes[4: 6], byteorder)
        sourcePoint = int.from_bytes(_bytes[6: 8], byteorder)
        destinationPoint = int.from_bytes(_bytes[8: 10], byteorder)
        dayOfGeneration = int.from_bytes(_bytes[10:12], byteorder)
        monthOfGeneration = int.from_bytes(_bytes[12:14], byteorder)
        yearOfGeneration = int.from_bytes(_bytes[14:16], byteorder)
        return ProductionOrderId(orderType, transportedGood, customerId, sourcePoint, destinationPoint,
                           dayOfGeneration, monthOfGeneration, yearOfGeneration)



class Frame:
    def addField(self, name, value):
        self.__dict__[name] = value
        return self


class FrameDescription:
    def __init__(self):
        self.fields = list()
        self.__initialize()

    def addField(self, field: FrameField):
        self.fields.append(field)
        return self

    def __initialize(self):
        fields = []
        for attrName in dir(self):
            attr = getattr(self, attrName)
            if isinstance(attr, FrameField):
                fields.append(attr)
        sortedFields = sorted(fields, key=lambda field: field.startingByte)
        for field in sortedFields:
            self.addField(field)

class FrameParser:
    def __init__(self, frameDescription: FrameDescription):
        self.__description = frameDescription

    def parse(self, data):
        frame = Frame()
        frameId = None
        for frameField in self.__description.fields:
            fieldName = frameField.name
            field = self.__parseField(data, frameField, frameId)
            if fieldName == 'id':
                frameId = field
            frame.addField(fieldName, field)
        return frame

    def __parseField(self, data, field, frameId):
        length = field.length
        if length == VARIABLE_LENGTH_MARK and frameId is not None:
            if frameId == 0:
                i = 0
                i += 1
            length = dataLengthByFrameId[frameId]
        tmp = copy.deepcopy(data[field.startingByte: (field.startingByte + length)])
        if field.type == bytes:
            return tmp
        return field.type.from_bytes(tmp, field.endianStyle)


class FrameBuilder:

    def __init__(self, frameDescription: FrameDescription):
        self.__description = frameDescription
        self.__fields = dict()

    def setFieldValue(self, name, value):
        self.__fields[name]=value
        return self

    def build(self):
        frameBytes = bytearray()
        for frameField in self.__description.fields:
            self.__buildField(frameField, frameBytes)
        return bytes(frameBytes)

    def __buildField(self, frameField, frameBytes):
        value = 0
        if frameField.name in self.__fields:
            value = self.__fields[frameField.name]
        length = frameField.length
        if length == VARIABLE_LENGTH_MARK:
            length = dataLengthByFrameId[self.__fields['id']]
        if type(value) != bytes:
            bytesValue = value.to_bytes(length, frameField.endianStyle)
        else:
            bytesValue = value
        frameBytes.extend(bytesValue)


class GenericFrameDescription(FrameDescription):
    beginningField = FrameField(startingByte=0, length=3, type=int, endianStyle='big', name='beginning')
    emptyField = FrameField(startingByte=3, length=1, type=int, endianStyle='big', name='empty')
    idField = FrameField(startingByte=4, length=2, type=int, endianStyle='big', name='id')
    statusField = FrameField(startingByte=6, length=2, type=int, endianStyle='big', name='status')
    timestampField = FrameField(startingByte=8, length=12, type=DtlDateTime, endianStyle='big', name='timestamp')
    numberField = FrameField(startingByte=20, length=2, type=int, endianStyle='big', name='number')
    versionField = FrameField(startingByte=22, length=2, type=int, endianStyle='big', name='version')
    lengthField = FrameField(startingByte=24, length=2, type=int, endianStyle='big', name='length')
    dataField = FrameField(startingByte=26, length=VARIABLE_LENGTH_MARK, type=bytes, endianStyle='big', name='data')
    endingField = FrameField(startingByte=VARIABLE_LENGTH_MARK, length=3, type=int, endianStyle='big', name='ending')

    def __init__(self):
        super().__init__()


# communication direction: MES -> TMS
class Frame5000Description(FrameDescription):
    timestampField = FrameField(startingByte=0, length=12, type=DtlDateTime, endianStyle='big', name='timestamp')
    productionOrderIdField = FrameField(startingByte=12, length=16, type=ProductionOrderId, endianStyle='big', name='productionOrderId')
    orderPriorityField = FrameField(startingByte=28, length=2, type=int, endianStyle='big', name='orderPriority')
    sourcePointIdField = FrameField(startingByte=30, length=2, type=int, endianStyle='big', name='sourcePointId')
    destinationPointIdField = FrameField(startingByte=32, length=2, type=int, endianStyle='big', name='destinationPointId')
    requiredOutputTimeField = FrameField(startingByte=34, length=12, type=DtlDateTime, endianStyle='big', name='requiredOutputTime')

    def __init__(self):
        super().__init__()


# communication direction: AGV -> TMS
class Frame6000Description(FrameDescription):
    timestampField = FrameField(startingByte=0, length=12, type=bytes, endianStyle='big', name='timestamp')
    generalSignalsField = FrameField(startingByte=12, length=2, type=bytes, endianStyle='big', name='generalSignals')
    safetySignalsField = FrameField(startingByte=14, length=6, type=bytes, endianStyle='big', name='safetySignals')
    LEDStatusField = FrameField(startingByte=20, length=4, type=bytes, endianStyle='big', name='ledStatus')
    exclusionStatusesField = FrameField(startingByte=24, length=2, type=bytes, endianStyle='big', name='exclusionStatuses')
    otherStatusesField = FrameField(startingByte=26, length=4, type=bytes, endianStyle='big', name='otherStatuses')
    weightSignalField = FrameField(startingByte=30, length=8, type=bytes, endianStyle='big', name='weightSignal')
    group1RightDriveSignalsField = FrameField(startingByte=38, length=4, type=bytes, endianStyle='big', name='rightDriveSignals')
    group2LeftDriveSignalsField = FrameField(startingByte=42, length=4, type=bytes, endianStyle='big', name='leftDriveSignals')
    group3BrakeSignalsField = FrameField(startingByte=46, length=2, type=bytes, endianStyle='big', name='brakeSignals')
    group4PinActuatorSignalsField = FrameField(startingByte=48, length=2, type=bytes, endianStyle='big', name='pinActuatorsSignals')
    group5LiftingPlateSignalField = FrameField(startingByte=50, length=2, type=bytes, endianStyle='big', name='liftingPlateSignals')
    alarmInformationField = FrameField(startingByte=52, length=10, type=bytes, endianStyle='big', name='alarmInformation')
    warningInformationField = FrameField(startingByte=62, length=4, type=bytes, endianStyle='big', name='warningInformation')
    messageInformationField = FrameField(startingByte=66, length=4, type=bytes, endianStyle='big', name='messageInformation')
    odometrySignalsField = FrameField(startingByte=70, length=12, type=bytes, endianStyle='big', name='odometrySignals')
    energySignalsField = FrameField(startingByte=82, length=16, type=bytes, endianStyle='big', name='energySignals')
    inclinationSignalsField = FrameField(startingByte=98, length=8, type=bytes, endianStyle='big', name='inclinationStatus')
    naturalNavigationSignalsField = FrameField(startingByte=106, length=36, type=bytes, endianStyle='big', name='naturalNavigationSignals')
    naturalNavigationCommandFeedbackField = FrameField(startingByte=142, length=20, type=bytes, endianStyle='big', name='naturalNavigationCommandFeedback')
    collaborativeRobotFeedbackField = FrameField(startingByte=162, length=10, type=bytes, endianStyle='big', name='collaborativeRobotFeedback')
    collaborativeRobotEnergySignalsField = FrameField(startingByte=172, length=16, type=bytes, endianStyle='big', name='collaborativeRobotEnergySignals')

    def __init__(self):
        super().__init__()


# communication direction: TMS -> AGV
class Frame6100Description(FrameDescription):
    timestampField = FrameField(startingByte=0, length=12, type=bytes, endianStyle='big', name='timestamp')
    generalSignalsControlField = FrameField(startingByte=12, length=2, type=bytes, endianStyle='big', name='generalSignalsControl')
    brakesControlField = FrameField(startingByte=14, length=2, type=bytes, endianStyle='big', name='brakesControl')
    actuatorControlField = FrameField(startingByte=16, length=2, type=bytes, endianStyle='big', name='actuatorControl')
    liftingPlateControlField = FrameField(startingByte=18, length=2, type=bytes, endianStyle='big', name='liftingPlateControl')
    LEDControlField = FrameField(startingByte=20, length=2, type=bytes, endianStyle='big', name='LEDControl')
    safetySignalsControlField = FrameField(startingByte=22, length=2, type=bytes, endianStyle='big', name='safetySignals')
    exclusionControlField = FrameField(startingByte=24, length=2, type=bytes, endianStyle='big', name='exclusionControl')
    customSignalControlField = FrameField(startingByte=26, length=40, type=bytes, endianStyle='big', name='customSignalControl')
    singleDrivesControlField = FrameField(startingByte=66, length=10, type=bytes, endianStyle='big', name='singleDrivesControl')
    manualControlField = FrameField(startingByte=76, length=6, type=bytes, endianStyle='big', name='manualControl')
    naturalNavigationCommandField = FrameField(startingByte=82, length=40, type=bytes, endianStyle='big', name='naturalNavigationCommand')
    collaborativeRobotCommandField = FrameField(startingByte=122, length=12, type=bytes, endianStyle='big', name='collaborativeRobotCommand')

    def __init__(self):
        super().__init__()
