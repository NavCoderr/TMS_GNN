from tms.test_utils.logger import Logger
import socket, sys, time, threading, random, signal, select
from tms.test_utils.sleepTimeFunction import sleepFunction
from frames_utils.frame import FrameBuilder, FrameParser, Frame6100Description, Frame6000Description, GenericFrameDescription


def getAcknowledgementFrame(working):
    value = 0
    if working:
        value = 1
    frame6000 = FrameBuilder(Frame6000Description()).setFieldValue('naturalNavigationCommandFeedback',
                                                                   value.to_bytes(20, 'big')).build()
    frame = FrameBuilder(GenericFrameDescription()).setFieldValue('id', 6000).setFieldValue('data', frame6000).build()
    return frame

def parseTmsRequest(frameBytes):
    genericFrameParser = FrameParser(GenericFrameDescription())
    frameData = genericFrameParser.parse(frameBytes).data
    try:
        feedbackBytes = FrameParser(Frame6100Description()).parse(frameData).naturalNavigationCommand
        return int.from_bytes(feedbackBytes, 'big')
    except Exception as e:
        return -1


working = False
workNumber = -1
interval = 1.0
executedTasks = 0


class AgvServer:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection = None
        self.__workingThread = None

    def connect(self, host, port):
        self.socket.bind((host, port))
        self.socket.listen(1)
        self.connection, _ = self.socket.accept()

    def run(self):
        while True:
            global working, logger
            if not working:
                ready = select.select([self.connection], [], [], 0)
                if ready[0]:
                    working = True
                    received = self.connection.recv(4096)
                    logger.logLine("Received: {}".format(received))
                    if len(received) > 0:
                        tmsRequest = parseTmsRequest(received)
                        if tmsRequest >= 0:
                            global workNumber
                            workNumber = int.from_bytes(received, 'big')
                            self.__createProcessingThread()
                        else:
                            print("Received invalid request")
            frame = getAcknowledgementFrame(working)
            self.connection.sendall(frame)
            time.sleep(0.5)

    def __createProcessingThread(self):
        self.__workingThread = threading.Thread(target=self.__processingThread)
        self.__workingThread.start()

    def __processingThread(self):
        global working, workNumber, interval, executedTasks, logger, tasksCountLogger
        workTime = sleepFunction(mean=interval)
        logger.logLine("Starting work {}, number: {} for: {}...".format(workNumber, executedTasks, workTime))
        time.sleep(workTime)
        executedTasks += 1
        tasksCountLogger.logLine("Executed tasks: {}".format(executedTasks))
        print("Executed tasks: {}".format(executedTasks))
        logger.logLine("\nDone")
        working = False



def onSigInt(signum, frame):
    global executedTasks, logger
    logger.logLine("SIGINT or SIGTERM handled")
    sys.exit(0)

tmp = sys.argv[1].split(':')
host, port, interval = tmp[0], int(tmp[1]), float(sys.argv[2])
s1 = signal.signal(signal.SIGINT, onSigInt)
s2 = signal.signal(signal.SIGTERM, onSigInt)

logger = Logger("test_agv_log_{}.txt".format(host).replace('.', '_'))
logger.logLine("Starting test agv on: {}:{}".format(host, port))
tasksCountLogger = Logger("agv_{}_tasks_count.txt".format(host).replace('.', '_'))
tasksCountLogger.logLine("Executed tasks: {}".format(executedTasks))

agvServer = AgvServer()
agvServer.connect(host, port)
agvServer.run()