from frames_utils.frame import FrameBuilder, Frame6100Description, GenericFrameDescription


class AgvFrameBuilder:
    def __init__(self):
        self.__nodeToVisit = None

    def startFrame(self):
        return self

    def withNodeToVisit(self, nodeNumber: int):
        self.__nodeToVisit = nodeNumber
        return self

    def consumeFrame(self):
        frame6100 = FrameBuilder(Frame6100Description()).setFieldValue('naturalNavigationCommand', self.__nodeToVisit.to_bytes(40, 'big')).build()
        frame = FrameBuilder(GenericFrameDescription()).setFieldValue('id', 6100).setFieldValue('data', frame6100).build()
        self.__nodeToVisit = None
        return frame