import json
from mes_adapter.test_utils.test_data import getTestFrame


class DataFormatter:
    def __init__(self):
        pass

    def getDataToSend(self, orderId, uniqueId):
        raise NotImplementedError()


class BinaryFormatter(DataFormatter):
    def __init__(self):
        super().__init__()

    def getDataToSend(self, orderId, uniqueId):
        return getTestFrame(orderId)


class JsonFormatter(DataFormatter):
    def __init__(self):
        super().__init__()

    def getDataToSend(self, orderId, uniqueId):
        data = { 'productionOrderId': orderId, 'id': uniqueId }
        return bytes(json.dumps(data), encoding='ASCII')