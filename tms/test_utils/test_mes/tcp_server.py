import threading, time, socket, sys, random, select


currentTask = -2
sleeping = False
sentTasksCount = 0


class TcpServer():
    def __init__(self, listener):
        super().__init__()
        self.__tasksLists = []
        self.__port = 0
        self.__host = 'localhost'
        self.__interval = 1.0
        self.__killed = False
        self.__serverThread = threading.Thread(target=self.__serverMain)
        self.__serverThread.daemon = True
        self.__listener = listener
        self.__server = None
        self.__sleepFunction = None
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__connection = None
        self.__dataFormatter = None

    def setDataFormatter(self, formatter):
        self.__dataFormatter = formatter

    def tasks(self):
        return self.__tasksLists

    def addTasks(self, tasks):
        self.__tasksLists.extend(tasks)

    def setPort(self, port):
        self.__port = port

    def setHost(self, host):
        self.__host = host

    def port(self):
        return self.__port

    def setInterval(self, interval):
        self.__interval = interval

    def interval(self):
        return self.__interval

    def start(self):
        self.__serverThread.start()

    def kill(self):
        self.__killed = True
        if self.__server is not None:
            self.__server.shutdown()

    def setSleepFunction(self, function):
        self.__sleepFunction = function

    def __askForSend(self):
        response = None
        ready = select.select([self.__connection], [], [], 0)
        if ready[0]:
            response = self.__connection.recv(1024)

        if response is not None:
            decodedResponse = bytes.decode(response, encoding='ASCII')
            return decodedResponse == "TMS_READY"
        return False

    def __askForAcceptedOrderId(self):
        response = None
        ready = select.select([self.__connection], [], [], 0)
        if ready[0]:
            response = self.__connection.recv(1024)

        if response is not None:
            decodedResponse = bytes.decode(response, encoding='ASCII')
            return int(decodedResponse)
        return -1

    def __serverMain(self):
        global currentTask, sleeping, sentTasksCount
        self.__socket.bind((self.__host, self.__port))
        self.__socket.listen(1)
        self.__connection, _ = self.__socket.accept()
        tmp = []
        while not self.__killed:
            oldStdOut = sys.stdout
            sys.stdout = None
            try:
                if len(tmp) == 0:
                    tmp = list(range(23, 36))
                    random.shuffle(tmp)
                if len(self.__tasksLists) > 0:
                    acceptedOrderId = -1
                    orderId = tmp.pop(0)
                    currentTask = self.__tasksLists.pop(0)
                    readyToSend = self.__askForSend()
                    while not readyToSend:
                        readyToSend = self.__askForSend()
                    sentTasksCount += 1
                    oldStdOut.write("Sending task: {}\n".format(currentTask))
                    frame = self.__dataFormatter.getDataToSend(orderId, sentTasksCount)
                    oldStdOut.write("{}\n".format(frame))
                    self.__connection.send(frame)
                    while acceptedOrderId != orderId:
                        acceptedOrderId = self.__askForAcceptedOrderId()
                        time.sleep(0.1)
                    time.sleep(0.1)
                else:
                    self.__connection.sendall(bytes())
                sys.stdout = oldStdOut
                # if self.__sleepFunction is None:
                #     time.sleep(self.__interval)
                # else:
                #     time.sleep(self.__sleepFunction(self.__interval))
            except Exception as e:
                sys.stdout = oldStdOut
                print(str(e))

