import json
from mes_adapter.request_parser import *


class JsonRequestParser(RequestParser):
    def parse(self, data) -> MesRequest:
        try:
            parsed = json.loads(data)
            return MesRequest(orderId=parsed['productionOrderId'], uniqueId=parsed['id'])
        except Exception:
            return MesRequest(orderId=-1, uniqueId=1)