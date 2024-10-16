import time, threading

from tcp_utils.client_utils import TcpClient
from agv_adapter.data_structures import *
from agv_adapter.request_builder import RequestBuilder


class REQUESTS:
    GET_AGVS_IDS = "GetAgvsIds"
    GET_AGV_STATUS = "GetAgvStatus"
    GO_TO_POINT = "GoToPoint"
    GO_TO_POINTS = "GoToPoints"


class AgvControllerClient:
    def __init__(self, agvControllerIp, agvControllerPort):
        self.__tcpClient = TcpClient(agvControllerIp, agvControllerPort)
        self.__lock = threading.Lock()
        self.__lastLockedThread = None

    def requestAgvsIds(self):
        request = RequestBuilder().startRequest(REQUESTS.GET_AGVS_IDS).finalize()
        response = self.__sendRequest(request)
        if response is not None:
            return agvIdsFromJson(response.decode('ASCII'))

    def requestAgvStatus(self, agvId):
        request = RequestBuilder().startRequest(REQUESTS.GET_AGV_STATUS).withAgvId(agvId).finalize()
        response = self.__sendRequest(request)
        if response is not None:
            return agvStatusFromJson(response.decode('ASCII'))

    def requestGoToPoint(self, agvId, point, taskId):
        request = RequestBuilder().startRequest(REQUESTS.GO_TO_POINT).withAgvId(agvId).withPoint(point).withTaskId(taskId).finalize()
        self.__sendRequest(request)

    def requestGoToPoints(self, agvId, points, taskId):
        if len(points) == 1:
            self.requestGoToPoint(agvId, points[0], taskId)
        else:
            request = RequestBuilder().startRequest(REQUESTS.GO_TO_POINTS).withAgvId(agvId).withPoints(points).withTaskId(taskId).finalize()
            self.__sendRequest(request)

    def connected(self):
        return self.__tcpClient.isConnected()

    def busy(self):
        return self.__lock.locked()

    def __waitForResponse(self):
        response = self.__tcpClient.readDataFromServer()
        for i in range(0, 100):
            if not self.connected():
                return None
            time.sleep(0.1)
            response = self.__tcpClient.readDataFromServer()
            if response is not None:
                return response
        return response

    def __sendRequest(self, request):
        if self.connected():
            if self.__lastLockedThread == threading.current_thread().name:
                raise Exception("Double-locked by the same thread!")

            with self.__lock:
                self.__lastLockedThread = threading.current_thread().name
                self.__tcpClient.sendDataToServer(request.encode('ASCII'))
                response = self.__waitForResponse()
                self.__lastLockedThread = ""
                return response
        return None
