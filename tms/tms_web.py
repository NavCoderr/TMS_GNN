import tempfile
from flask import Flask, request, render_template, abort, send_file, redirect
from dataclasses import dataclass
from composition_root import CompositionRoot, TmsInitInfo, QueueObserver
import logging

@dataclass
class AGV:
    name: str
    tasks: list
    state: str

@dataclass
class AGVTask:
    id: str
    state: str

@dataclass
class TaskView:
    number: str
    state: str


class WebQueueObserver(QueueObserver):
    def __init__(self):
        self.agvs = []
        self.tasks = []
        self.pendingTasks = []
        self.cost = 0
        self.length = 0

    def probeQueueState(self, queue, executorsViews, timePoint):
        self.reset()
        if queue.cost() != -1:
            self.cost = str(round(queue.cost()))
        else:
            self.cost = "Never"
        for executorView in executorsViews:
            agvTasks = []
            pathPoint = executorView.pathPoint()
            assignedPath = executorView.assignedPath()
            for i in range(0, len(assignedPath)):
                state = 'waiting'
                if i == pathPoint:
                    state = 'executing'
                if i < pathPoint:
                    state = 'executed'
                agvTasks.append(AGVTask(id=assignedPath[i], state=state))
            self.agvs.append(AGV(executorView.executorId(), agvTasks, executorView.state()))
        for task in queue.tasksList():
            self.tasks.append(TaskView(task.taskNumber(), "optimized"))
        for task in queue.pendingTasksList():
            self.tasks.append(TaskView(task.taskNumber(), "pending"))
        self.length = len(self.tasks)

    def reset(self):
        self.agvs = []
        self.tasks = []
        self.pendingTasks = []
        self.cost = 0
        self.length = 0


class WebTms:
    def __init__(self):
        self.__tms = CompositionRoot()
        self.queueObserver = WebQueueObserver()
        self.tmsInitInfo = None

    def start(self, initInfo : TmsInitInfo):
        if not self.isRunning():
            self.tmsInitInfo = initInfo
            self.__tms.initialize(self.tmsInitInfo)
            self.__tms.start()

    def shutdown(self):
        self.__tms.shutdown()
        self.__tms = None

    def isRunning(self):
        print(self.__tms.isRunning())
        return self.__tms is not None and self.__tms.isRunning()

    def mesStatus(self):
        connected = False
        if self.__tms:
            connected = self.__tms.isMesConnected()
        if connected:
            return "Connected"
        return "Disconnected"

    def simulationMesStatus(self):
        connected = False
        if self.__tms:
            connected = self.__tms.isSimulationMesConnected()
        if connected:
            return "Connected"
        return "Disconnected"

    def agvHubStatus(self):
        if self.__tms and self.__tms.isAgvHubConnected():
            return "Connected"
        return "Disconnected"

    def addFile(self, file):
        self.__tms.registerFile(file)


app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

tms = WebTms()

@app.route('/')
def index():
    disabledTag = ''
    #if False:
    #    disabledTag = 'disabled'

    return render_template('index.html', runDisabledTag=disabledTag)

def getFileFromRequest(request, filename):
    if filename not in request.files:
        return ''
    file = request.files[filename]
    if file.filename == '':
        return ''
    return file


@app.route('/run', methods=['GET', 'POST'])
def run():
    if request.method == 'POST':
        if request.form.get('Run') == 'Run':
            if not tms.isRunning():
                topologyDescriptionFile = getFileFromRequest(request, 'graphDescription')
                mesMappingFile = getFileFromRequest(request, 'mesMappingFile')
                tms.addFile(topologyDescriptionFile)
                tms.addFile(mesMappingFile)
                topologyDescriptionFile = topologyDescriptionFile.name
                mesMappingFile = mesMappingFile.name

                mesConnectionString = request.form.get('mesConnectionString').split(':')
                simulationMesConnectionString = request.form.get('simulationMesConnectionString').split(':')
                agvHubConnectionString = request.form.get('agvHubConnectionString').split(':')
                tmsInitInfo = TmsInitInfo(topologyDescriptionPath=topologyDescriptionFile,
                                          mesIp=mesConnectionString[0], mesPort=int(mesConnectionString[1]),
                                          agvControllerIp=agvHubConnectionString[0], agvControllerPort=int(agvHubConnectionString[1]),
                                          mesTasksMappingPath=mesMappingFile, queueObserver=tms.queueObserver,
                                          simulationMesIp=simulationMesConnectionString[0], simulationMesPort=int(simulationMesConnectionString[1]),)
                tms.start(tmsInitInfo)

    if tms.tmsInitInfo is None:
        return redirect('/')

    mesConnectionString = "{}:{}".format(tms.tmsInitInfo.mesIp, tms.tmsInitInfo.mesPort)
    agvHubConnectionString = "{}:{}".format(tms.tmsInitInfo.agvControllerIp, tms.tmsInitInfo.agvControllerPort)
    simulationMesConnectionString = "{}:{}".format(tms.tmsInitInfo.simulationMesIp, tms.tmsInitInfo.simulationMesPort)

    return render_template('tms.html',
                           mesStatus=tms.mesStatus(),
                           simulationMesStatus=tms.simulationMesStatus(),
                           agvHubStatus=tms.agvHubStatus(),
                           mesConnectionString=mesConnectionString,
                           agvHubConnectionString=agvHubConnectionString,
                           simulationMesConnectionString=simulationMesConnectionString,
                           tasksQueue=tms.queueObserver.tasks,
                           agvs=tms.queueObserver.agvs,
                           cost=tms.queueObserver.cost,
                           length=tms.queueObserver.length)
