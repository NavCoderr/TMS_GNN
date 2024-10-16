from frames_utils.frame import *
from mes_adapter.request_parser import *


class MesFrameBuilder(FrameBuilder):
    def __init__(self):
        super().__init__(GenericFrameDescription())


class MesFrameParser(RequestParser):
    def __init__(self):
        self.__data = None

    def parse(self, data):
        self.__data = data
        return self.__parse()

    def __parse(self):
        try:
            genericFrameParser = FrameParser(GenericFrameDescription())
            frameData = genericFrameParser.parse(self.__data).data
            frame = FrameParser(Frame5000Description()).parse(frameData)
            return MesRequest(orderId=frame.productionOrderId.orderType, uniqueId=1)
        except Exception as e:
            print(e)
            return MesRequest(orderId=-1, uniqueId=1)

    def __validate(self, totalLength):
        pass
        # todo: uncomment in production
        # if len(self.__data) != totalLength:
        #     raise Exception("Invalid frame!")

