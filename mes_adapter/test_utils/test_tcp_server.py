import socketserver
from mes_adapter.test_utils.test_data import getTestFrame


class TestTcpHandler(socketserver.BaseRequestHandler):

    def handle(self):
        self.request.sendall(getTestFrame())


host, port = 'localhost', 1234

with socketserver.TCPServer((host, port), TestTcpHandler) as server:
    server.serve_forever()