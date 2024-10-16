import unittest
from mes_adapter.test_utils.test_data import getTestFrame
from frames_utils.frame import *


class TasksSchedulingTests(unittest.TestCase):


    def test_parsesGenericFrameCorrectly(self):
        parser = FrameParser(GenericFrameDescription())
        parsedFrame = parser.parse(getTestFrame())
        self.assertEqual(parsedFrame.id, 5000)
        self.assertEqual(parsedFrame.beginning, 3950171)
        self.assertEqual(parsedFrame.empty, 0)
        self.assertEqual(parsedFrame.ending, 0)
        self.assertEqual(parsedFrame.length, 70)
        self.assertEqual(parsedFrame.number, 2600)
        self.assertEqual(parsedFrame.status, 1)
        self.assertEqual(parsedFrame.version, 1)
        dtlTimeStamp = parsedFrame.timestamp
        self.assertEqual(dtlTimeStamp.year, 1970)
        self.assertEqual(dtlTimeStamp.month, 1)
        self.assertEqual(dtlTimeStamp.day, 1)
        self.assertEqual(dtlTimeStamp.dayOfWeek, 5)
        self.assertEqual(dtlTimeStamp.hour, 0)
        self.assertEqual(dtlTimeStamp.minute, 0)
        self.assertEqual(dtlTimeStamp.second, 0)
        self.assertEqual(dtlTimeStamp.millisecond, 0)

    def test_parsesFrame5000Correctly(self):
        parsedGenericFrame = FrameParser(GenericFrameDescription()).parse(getTestFrame())
        parsed5000Frame = FrameParser(Frame5000Description()).parse(parsedGenericFrame.data)
        self.assertEqual(parsed5000Frame.productionOrderId, 23)
        self.assertEqual(parsed5000Frame.orderPriority, 20)
        self.assertEqual(parsed5000Frame.sourcePointId, 48)
        self.assertEqual(parsed5000Frame.destinationPointId, 2023)
        timestamp = parsed5000Frame.timestamp
        self.assertEqual(timestamp.year, 525)
        self.assertEqual(timestamp.month, 2)
        self.assertEqual(timestamp.day, 23)
        self.assertEqual(timestamp.dayOfWeek, 21)
        self.assertEqual(timestamp.hour, 29)
        self.assertEqual(timestamp.minute, 58)
        self.assertEqual(timestamp.second, 147)
        self.assertEqual(timestamp.millisecond, 1420820481)
        requiredOutput = parsed5000Frame.requiredOutputTime
        self.assertEqual(requiredOutput.year, 770)
        self.assertEqual(requiredOutput.month, 5)
        self.assertEqual(requiredOutput.day, 12)
        self.assertEqual(requiredOutput.dayOfWeek, 11)
        self.assertEqual(requiredOutput.hour, 10)
        self.assertEqual(requiredOutput.minute, 0)
        self.assertEqual(requiredOutput.second, 0)
        self.assertEqual(requiredOutput.millisecond, 23878)


    def test_buildsFrameCorrectly(self):
        frameBuilder = FrameBuilder(GenericFrameDescription())
        frameBuilder.setFieldValue('id', 5000).setFieldValue('beginning', 3950171).setFieldValue('data', 0).\
            setFieldValue('empty', 0).setFieldValue('ending', 0).setFieldValue('length', 70).setFieldValue('number', 2600).\
            setFieldValue('status', 1).setFieldValue('timestamp', DtlDateTime(2023, 9, 22, 5, 14, 0, 0, 12)).setFieldValue('version', 1)
        frameBytes = frameBuilder.build()
        parsedFrame = FrameParser(GenericFrameDescription()).parse(frameBytes)
        self.assertEqual(parsedFrame.id, 5000)
        self.assertEqual(parsedFrame.beginning, 3950171)
        self.assertEqual(parsedFrame.data, bytes(32))
        self.assertEqual(parsedFrame.empty, 0)
        self.assertEqual(parsedFrame.ending, 0)
        self.assertEqual(parsedFrame.length, 70)
        self.assertEqual(parsedFrame.number, 2600)
        self.assertEqual(parsedFrame.status, 1)
        self.assertEqual(parsedFrame.version, 1)
        dtlTimeStamp = parsedFrame.timestamp
        self.assertEqual(dtlTimeStamp.year, 2023)
        self.assertEqual(dtlTimeStamp.month, 9)
        self.assertEqual(dtlTimeStamp.day, 22)
        self.assertEqual(dtlTimeStamp.dayOfWeek, 5)
        self.assertEqual(dtlTimeStamp.hour, 14)
        self.assertEqual(dtlTimeStamp.minute, 0)
        self.assertEqual(dtlTimeStamp.second, 0)
        self.assertEqual(dtlTimeStamp.millisecond, 12)

    def test_buildBinarlyCompatibleFrame(self):
        frame5000Builder = FrameBuilder(Frame5000Description())
        frame5000Builder.setFieldValue('productionOrderId', 23).setFieldValue('orderPriority', 20).\
            setFieldValue('sourcePointId', 48).setFieldValue('destinationPointId', 2023).\
            setFieldValue('timestamp', DtlDateTime(525, 2, 23, 21, 29, 58, 147, 1420820481)).\
            setFieldValue('requiredOutputTime', DtlDateTime(770, 5, 12, 11, 10, 0, 0, 23878))
        mesFrameBuilder = FrameBuilder(GenericFrameDescription())
        mesFrameBuilder.setFieldValue('id', 5000).setFieldValue('beginning', 3950171).setFieldValue('data', frame5000Builder.build()).\
            setFieldValue('empty', 0).setFieldValue('ending', 0).setFieldValue('length', 70).setFieldValue('number', 2600).\
            setFieldValue('status', 1).setFieldValue('timestamp', DtlDateTime(1970, 1, 11, 5, 0, 0, 0, 0)).setFieldValue('version', 1)
        frameBytes = mesFrameBuilder.build()
        goldenBytes = getTestFrame()
        self.assertEqual(len(frameBytes), len(goldenBytes))
        self.assertEqual(frameBytes, goldenBytes)