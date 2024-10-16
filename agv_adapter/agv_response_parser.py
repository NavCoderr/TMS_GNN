from frames_utils.frame import FrameParser, Frame6000Description, GenericFrameDescription


class AgvResponseParser:
    def __init__(self, frame):
        self.__frame = frame

    def getNaturalNavigationCommandFeedback(self):
        try:
            genericFrameParser = FrameParser(GenericFrameDescription())
            frameData = genericFrameParser.parse(self.__frame).data
            feedbackBytes = FrameParser(Frame6000Description()).parse(frameData).naturalNavigationCommandFeedback
            return int.from_bytes(feedbackBytes, 'big')
        except Exception as e:
            return -1
