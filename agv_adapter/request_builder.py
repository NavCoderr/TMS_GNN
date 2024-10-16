import json


class REQUESTS_KEYS:
    REQUEST_ID = "id"
    AGV_ID = "agv_id"
    POINTS = "points"
    POINT = "point"
    TASK_ID = "task_id"


class RequestBuilder:
    def __init__(self):
        self.__request = {}

    def startRequest(self, requestId):
        self.__request[REQUESTS_KEYS.REQUEST_ID] = requestId
        return self

    def withAgvId(self, agvId):
        self.__request[REQUESTS_KEYS.AGV_ID] = agvId
        return self

    def withPoints(self, points):
        self.__request[REQUESTS_KEYS.POINTS] = points
        return self

    def withPoint(self, point):
        self.__request[REQUESTS_KEYS.POINT] = point
        return self

    def withTaskId(self, taskId):
        self.__request[REQUESTS_KEYS.TASK_ID] = taskId
        return self

    def finalize(self):
        return json.dumps(self.__request)